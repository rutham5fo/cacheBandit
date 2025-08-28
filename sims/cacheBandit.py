import math
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matplotlib.pyplot as plt
from queue import Queue as Q
import sys
import csv
import more_itertools as mit
import random
from contextlib import redirect_stdout
import datetime

"""
NOTES:  -> Up switch and deallocator are merged into one module, called the refresh unit.
        -> The refresh Unit is triggered each time the warp changes pages. CDC is through
           a single vld bit which toggles on a page change. Hence, the requirement of twice
           IPC for reclaim size, so that the slower cacheBandit clk can still capture this flag.
"""

# Consts
life_MAX    = 9999

# Physical memory
shmem_line = 16
shmem_assoc = 4                            # Associativity per set of cache
shmem_sets = 32                             # No. of sets in cache
shmem_size = shmem_assoc * shmem_sets
page_size = 32                              # !! Page_size must be >= than 4 * IPC (2 * reclaim size) count, for safe fast->slow CDC !!
pages = int(shmem_size/page_size)           # !! Pages >= 2 for safe slow->fast CDC !!

# Worker and warp related params
workers = 4                 # No. of cores/warps
contexts = 4                # No. of warp contexts TDM'd by worker @ Freq_cacheBandit

# CacheBandit constants
freq_cacheBandit = 100 #MHz
freq_PE = contexts * freq_cacheBandit #MHz
ipc = int(math.ceil(freq_PE / freq_cacheBandit))
bytes_per_line = 128

# Stats related
ext_mem_access_penalty = 8                  # External memory latency


class bandit:

    def __init__ (self, block_id, page_size, pages, ipc=1):
        # CacheBandit
        self.ipc = ipc
        self.bid = block_id
        # Page
        self.pages = pages
        self.page_size = page_size          # Total number of lines in a page
        self.reclaim_size = 2 * self.ipc    # reclaim twice IPC for fast to slow domain crossing
        self.refresh_threshold = self.page_size - self.ipc
        self.page_life = [0 for _ in range(self.pages)]
        self.cur_sol = [self.page_size-self.reclaim_size for _ in range(self.pages)]
        self.cur_page = 0
        self.next_page = 0
        self.prev_page = 0
        self.vmap = [[v for v in range(self.page_size)] for _ in range(self.pages)]
        self.pmap = [[p for p in range(self.page_size)] for _ in range(self.pages)]
        self.smap = [[[0, 0, 0] for _ in range(self.page_size)] for _ in range(self.pages)]         # [set, line, vld]
        self.rpmap = [p for p in range(self.page_size)]
        self.rsmap = [[0, 0, 0] for _ in range(self.page_size)]
        self.commit_base_ptr = 0
        # Stats
        self.alloc_cnt = 0
        self.dealloc_cnt = 0
        # initialize
        self.max_hops = (self.pages * int(self.page_size / self.reclaim_size)) - 1
        self.init = 1
        self.hop_count = 0
        self.page_life[self.cur_page] = life_MAX
        self.next_page = self.cur_page + 1

    def print_stats (self):
        # Compute stats
        peak_page_util, avg_page_util = 0, 0
        for smap in self.smap:
            page_util = 0
            for line in smap:
                page_util += line[2]
            avg_page_util += page_util
            if (page_util >= peak_page_util):
                peak_page_util = page_util
        avg_page_util = avg_page_util / (self.page_size * self.pages)
        peak_page_util = peak_page_util / self.page_size

        print (f'{bandit.print_stats.__name__} ||| ------ BANDIT[{self.bid}] STATS BEGIN ------ ')
        print (f'{bandit.print_stats.__name__} ||| Allocs_called = {self.alloc_cnt}, Deallocs_called = {self.dealloc_cnt} | Average Page Utilization = {avg_page_util}, Peak Page Utilization = {peak_page_util}')
        print (f'{bandit.print_stats.__name__} ||| ---------- BANDIT END ---------- ')

        return avg_page_util, peak_page_util

    def reset_stats (self):
        self.alloc_cnt = 0
        self.dealloc_cnt = 0

    def reset_page (self):
        self.init = 1
        self.hop_count = 0
        self.cur_sol = [self.page_size-self.reclaim_size for _ in range(self.pages)]
        self.cur_page = 0
        self.next_page = self.cur_page + 1
        self.vmap = [[v for v in range(self.page_size)] for _ in range(self.pages)]
        self.pmap = [[p for p in range(self.page_size)] for _ in range(self.pages)]
        self.smap = [[[0, 0, 0] for _ in range(self.page_size)] for _ in range(self.pages)]
        self.rpmap = [p for p in range(self.page_size)]
        self.rsmap = [[0, 0, 0] for _ in range(self.page_size)]
        self.commit_base_ptr = 0

    def get_alloc_cnt (self):
        return self.alloc_cnt
    
    def get_dealloc_cnt (self):
        return self.dealloc_cnt
    
    """
    Used by warp to perform page touch
    Exposed in hardware as part of the buffer (CDC)
    """
    def get_cur_page (self):
        return self.cur_page
    
    def page_touch (self, page):
        self.page_life[page] = life_MAX

    def page_lru (self):
        base_life = life_MAX
        comparator = []
        for pid, life in enumerate(self.page_life):
            comparator.append([life, pid])
            #print (f'{bandit.page_lru.__name__} [{self.bid}] ||| PAGE LRU || Page = {pid}, life = {life}, base_life = {base_life}', flush=True)
        lru_meta = min(comparator)
        lru_id = lru_meta[1]
        #print (f'{bandit.page_lru.__name__} [{self.bid}] ||| PAGE LRU || pglru = lru_id')
        return lru_id

    def age_page (self):
        for x in range(self.pages):
            if (self.page_life[x] > 0):
                self.page_life[x] -= 1
    
    def dealloc_refresh (self, page):
        # Fall back to FIFO reclaim policy if seed is none
        beg = self.reclaim_size - 1
        end = self.page_size - 1
        vptr = beg if (self.init and self.hop_count <= self.max_hops-self.pages) else random.randint(beg, end)
        this_page = page
        # Copy page context
        self.rpmap = self.pmap[this_page]
        self.rsmap = self.smap[this_page]
        # Enque for marking and rotate page
        base_ptr = vptr-self.reclaim_size+1
        for x in range(self.reclaim_size):
            #print (f'{bandit.dealloc_refresh.__name__} [{self.bid}] | Page[{this_page}] ||| iter = {x} | base_ptr = {base_ptr}, vptr = {vptr}, reclaim_size = {self.reclaim_size}', flush=True)
            try:
                cache_ptr = self.smap[this_page][base_ptr]
            except IndexError as e:
                #print (f'{bandit.dealloc_refresh.__name__} [{self.bid}] | Page[{this_page}] ||| {e} | base_ptr = {base_ptr}, vptr = {vptr}, reclaim_size = {self.reclaim_size}', flush=True)
                sys.exit(1)
            #print (f'{bandit.dealloc_refresh.__name__} [{self.bid}] | cache_ptr = {cache_ptr}')
            # Rotate
            tpmap = self.pmap[this_page][base_ptr]
            tsmap = self.smap[this_page][base_ptr]
            shift_pmap = self.rpmap[1:] + self.rpmap[:1]
            shift_smap = self.rsmap[1:] + self.rsmap[:1]
            shift_pmap[-1] = tpmap
            shift_smap[-1] = tsmap
            #print (f'{bandit.dealloc_refresh.__name__}[{self.bid}] | Page[{this_page}] ||| Rpmap before = {self.rpmap} | len = {len(self.rpmap)}', flush=True)
            self.rpmap = shift_pmap
            self.rsmap = shift_smap
            #print (f'{bandit.dealloc_refresh.__name__}[{self.bid}] | Page[{this_page}] ||| Rpmap after = {self.rpmap} | len = {len(self.rpmap)}', flush=True)
            # Next Step
            self.commit_base_ptr = base_ptr
            base_ptr += 1

        # Fill regs
        #self.next_page = this_page
        #print (f'{bandit.dealloc_refresh.__name__}[{self.bid}] |  next_page = {self.next_page}')

    def dealloc_commit (self, page):
        if (self.commit_base_ptr != 0):
            # Merge rmap with pmap
            delim = self.commit_base_ptr - self.reclaim_size + 1
            #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| commit_ptr = {self.commit_base_ptr}, cur_sol = {self.cur_sol[page]}, pmap = {self.pmap[page]}', flush=True)
            if (delim != 0):
                self.rpmap[0:delim] = self.pmap[page][0:delim]
                self.rsmap[0:delim] = self.smap[page][0:delim]
            #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| Post-shift', flush=True)
            #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| pmap = {self.rpmap}', flush=True)
            self.cur_sol[page] -= self.reclaim_size
            self.dealloc_cnt += 1
            # Refresh page
            self.pmap[page] = self.rpmap
            self.smap[page] = self.rsmap
            #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| cur_sol = {self.cur_sol[page]}, next_alloc = {self.pmap[page][self.cur_sol[page]]}', flush=True)
    
    def clean (self):
        pg_ptr = self.cur_sol[self.cur_page]
        # Get LRU page
        init_page_sel = lambda x: 0 if (x == self.pages-1) else self.cur_page+1
        page = init_page_sel(self.cur_page) if (self.init) else self.page_lru()
        # Check page utilization
        #print (f'{bandit.clean.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Current Page ptr = {pg_ptr}, dealloc_refresh_threshold = {self.refresh_threshold} | Page_lru = {page}', flush=True)
        if (pg_ptr == self.refresh_threshold):
            self.dealloc_commit(self.prev_page)
            self.dealloc_refresh(self.cur_page)
            self.prev_page = self.cur_page
            self.next_page = page
            #print (f'{bandit.clean.__name__}[{self.bid}] |  next_page = {self.next_page}')
        elif (pg_ptr == self.page_size):
            #print (f'{bandit.clean.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Page switch. Next page = {self.next_page} | Hop_count = {self.hop_count}, init = {self.init}, max_hops = {self.max_hops}', flush=True)
            self.cur_page = self.next_page
            self.hop_count += 1 if (self.init) else 0
            self.init = 0 if (self.hop_count == self.max_hops) else self.init
    
    def allocate (self, set_sel, line_id):
        # Check for page overflow
        if (self.cur_sol[self.cur_page] == self.page_size):
            print (f'{bandit.allocate.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Page-Overflow ! | cur_sol = {self.cur_sol[self.cur_page]}, page_size = {self.page_size}')
            raise IndexError("Page-Overflow!")
        sol_sel = self.cur_sol[self.cur_page]
        prev_sol = self.pmap[self.cur_page][sol_sel]
        prev_set = self.smap[self.cur_page][sol_sel][0]
        prev_line = self.smap[self.cur_page][sol_sel][1]
        prev_vld = self.smap[self.cur_page][sol_sel][2]
        self.smap[self.cur_page][sol_sel][0] = set_sel
        self.smap[self.cur_page][sol_sel][1] = line_id
        self.smap[self.cur_page][sol_sel][2] = 1
        self.cur_sol[self.cur_page] += 1
        #print (f'{bandit.allocate.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Allocating line {prev_sol} | new_sol = {self.cur_sol[self.cur_page]} | set_sel = {set_sel}, line_id = {line_id}', flush=True)
        self.alloc_cnt += 1
        
        # Housekeeping
        self.clean()

        return (prev_sol, self.cur_page, prev_set, prev_line, prev_vld)

class warp:

    # Common
    __obj = []
    __L2_hit = 0

    def __init__ (self, warp_id, virtual_line, virtual_assoc, virtual_sets, page_size, pages, ipc=1):
        # warp
        self.tid = warp_id
        self.vline = virtual_line
        self.vassoc = virtual_assoc
        self.vsets = virtual_sets
        self.vlru = [int(math.ceil(x%self.vassoc)) for x in range(self.vsets)]
        self.vcache = [[[None, (None, None), 0, 0] for _ in range(self.vassoc)] for _ in range(self.vsets)]  # [Key, (phy_ptr, page), lifetime, valid] per assoc
        self.buf = [None, (None, None)]                 # [Key, (phy_ptr, page)]
        self.cb = bandit(self.tid, page_size, pages, ipc)
        # Stats
        self.ext_hit = 0
        self.int_hit = 0
        self.cycle = 0
        # Common
        self.append_obj(self)

    @classmethod
    def append_obj (cls, obj):
        if (obj is not None):
            cls.__obj.append(obj)
    
    @classmethod
    def reset_common (cls):
        cls.__obj = []
        cls.__L2_hit = 0
    
    @classmethod
    def get_L2_stats (cls):
        return cls.__L2_hit

    @classmethod
    def find_L2 (cls, set_sel, key, tid):
        found = 0
        ptr = None
        page = None
        for t in cls.__obj:
            if (t.tid != tid and found == 0):
                s = t.vcache[set_sel]
                at = 0
                for line_id, line in enumerate(s):
                    if (line[0] == key and line[3]):
                        #print (f'{warp.find_L2.__name__} [{tid}] ||| Found key [{key}] in L2 at location [{set_sel}][{line_id}], in warp [{t.tid}] | line = {line}', flush=True)
                        found = 1
                        ptr = line[1][0]
                        page = line[1][1]
                        # Function only cares about finding in L2.
                        # Assuming the external memory access time is greater than
                        # the time taken by requesting worker to copy the line
                        # into its own cache line buffer (self.buf). The line will
                        # be copied by the time it is overwritten.
                        cls.__L2_hit += 1
                        break

        return found, ptr, page

    def print_stats (self):
        # Compute stats
        total_access = self.ext_hit + self.int_hit
        hit_rate = self.int_hit / total_access
        miss_rate = self.ext_hit / total_access
        
        print (f'{warp.print_stats.__name__} ||| ------ WARP[{self.tid}] STATS BEGIN ------ ')
        print (f'{warp.print_stats.__name__} ||| Total accesses = {total_access} | mem_ext_hit = {self.ext_hit}, mem_int_hit = {self.int_hit} | hit_rate = {hit_rate}, miss_rate = {miss_rate}')
        print (f'{warp.print_stats.__name__} ||| ---------- WARP END ----------')

        avg_page_util, peak_page_util = self.cb.print_stats()

        print (f'', flush=True)

        return total_access, self.ext_hit, self.int_hit, self.cycle, avg_page_util, peak_page_util

    def reset_stats (self):
        self.ext_hit = 0
        self.int_hit = 0
        self.cycle = 0
    
    def reset_bandit (self):
        # Reset Bandit
        self.cb.reset_stats()
        self.cb.reset_page()
    
    def invalidate (self, alloc, set_sel, line, vld):
        #if (alloc and vld and self.disable == 0):
        if (alloc and vld):
            #print (f'{warp.invalidate.__name__} ||| Invalidating set[{set_sel}], line[{line}] = {self.vcache[set_sel][line]}', flush=True)
            self.vcache[set_sel][line][3] = 0

    def find_lru (self, set_sel, vset):
        base_life = life_MAX
        comparator = []
        for line_id, line in enumerate(vset):
            line_life = line[2]
            comparator.append([line_life, line_id])
            #print (f'{warp.find_lru.__name__} [{self.tid}] ||| LRU || Frame = {line[0]}, phy_ptr = {line[1]}, line_life = {line_life}, base_life = {base_life}', flush=True)
        lru_meta = min(comparator)
        lru_id = lru_meta[1]
        self.vlru[set_sel] = lru_id
        #print (f'{warp.find_lru.__name__} [{self.tid}] ||| LRU || list = {self.vlru}')
    
    def age_cache(self):
        for sid, s in enumerate(self.vcache):
            for lid, line in enumerate(s):
                if (line[2] > 0):
                    line[2] -= 1
                    #if (line[2] < 0):
                    #    print (f'{warp.age_cache.__name__} ||| Age turns negative!')
                    #    sys.exit(1)
    
    def show_cache(self, set_sel):
        print (f'{warp.show_cache.__name__} [{self.tid}] ||| Dumping Cache state:\n')
        for set_id, s in enumerate(self.vcache):
            if (set_id == set_sel or set_sel == self.vsets):
                for line_id, line in enumerate(s):
                    print (f'{warp.show_cache.__name__} [{self.tid}] ||| Set[{set_id}] || Line[{line_id}] = {line}', flush=True)
    
    def find_buf (self, key):
        found = 0
        if (self.buf[0] == key):
            #print (f'{warp.find_buf.__name__} [{self.tid}] ||| Found key [{key}] in Buffer', flush=True)
            found = 1
        return found
    
    def find_L1 (self, set_sel, key):
        found_L1 = 0
        found_L2 = 0
        ev_lid = self.vlru[set_sel]         # eviction line id
        s = self.vcache[set_sel]            # current set
        at = ev_lid
        ptr = s[at][1][0]
        page = s[at][1][1]
        life = s[at][2]
        vld = s[at][3]
        evict_line = [s[at][0], (ptr, page), life, vld]    # Current line to evict
        alloc = 0
        invld_set = 0
        invld_line = 0
        invld_vld = 0
        l2_ptr = 0
        l2_page = 0

        # Check L1
        for line_id, line in enumerate(s):
            if (line[0] == key and line[3]):
                #print (f'{warp.find_L1.__name__} [{self.tid}] ||| Found key [{key}] in L1 at location [{set_sel}][{line_id}] | line = {line}', flush=True)
                found_L1 = 1
                at = line_id
                ptr = line[1][0]
                page = line[1][1]
                life = life_MAX
                vld = line[3]
                self.cb.page_touch(page)
        
        #print (f'{warp.find_L1.__name__} [{self.tid}] ||| found = {found}, at = {at}, ptr = {ptr}, page = {page}, life = {life}, vld = {vld} | alloc = {alloc}', flush=True)

        # Check L2
        if (found_L1 == 0):
            found_L2, l2_ptr, l2_page = self.find_L2(set_sel, key, self.tid)

        # Allocate/Insert
        if (found_L1 == 0 and found_L2 == 0):
            if (vld):
                alloc = 0
                ptr, page, invld_set, invld_line, invld_vld = ptr, page, invld_set, invld_line, invld_vld
                self.cb.page_touch(page)
            else:
                alloc = 1
                cur_page = self.cb.get_cur_page()
                self.cb.page_touch(cur_page)
                ptr, page, invld_set, invld_line, invld_vld = self.cb.allocate(set_sel, at)
            life = life_MAX
            vld = 1
            # Evict current LRU line, and insert new line
            #print (f'{warp.find_L1.__name__} [{self.tid}] ||| Inserting key [{key}] in L1 at location [{set_sel}][{ev_lid}] | evicted_line = {evict_line}, inserted_line = {[key, (ptr, page), life, vld]} | alloc = {alloc}', flush=True)
        elif (found_L2):
            ptr, page = l2_ptr, l2_page
            self.buf = [key, (ptr, page)]
        
        insert_line = [key, (ptr, page), life, vld]

        # Insert line
        s[at] = insert_line

        # Invalidate line
        self.invalidate(alloc, invld_set, invld_line, invld_vld)

        # Update LRU of current set
        self.find_lru(set_sel, s)

        return found_L1, found_L2
        
    def strm (self, taddr):
        found_buf = 0
        found_L1 = 0
        found_L2 = 0
        # Age cache
        self.age_cache()
        # Age page
        self.cb.age_page()
        # Read address from context stream
        addr = int(taddr)
        vframe = int(math.floor(addr / self.vline))
        vset_sel = int(math.floor(vframe % self.vsets))
        #print (f'{warp.strm.__name__} ||| ------- RUN warp [{self.tid}] ------- || strm_addr = {taddr} | Key = {vframe} | Set = {vset_sel}', flush=True)
        
        # Find addr in buffer
        found_buf = self.find_buf(vframe)
        # Find addr in Cache
        found_L1, found_L2 = self.find_L1(vset_sel, vframe)

        # Increament Penalty/load cost + Stats
        if (found_L1 or found_buf):
            self.int_hit += 1
            self.cycle += 1
        elif (found_L2):
            # L2 hit increamented in classmethod
            self.cycle += contexts
        else:
            self.ext_hit += 1
            self.cycle += ext_mem_access_penalty

def scheduler (ctxt, wid, worker, disable=0):
    for id, elem in enumerate(ctxt):
        wsel = 0 if (disable or wid is None) else int(wid[id])
        taddr = int(elem)
        worker[wsel].strm(taddr)

def workers_test (ctxt_fname, vline, vassoc, vsets, page_size, pages, ipc):
    total_access = [0 for _ in range(workers)]
    int_hits = [0 for _ in range(workers)]
    ext_hits = [0 for _ in range(workers)]
    load_dist = [0 for _ in range(workers)]
    avg_page_util = [0 for _ in range(workers)]
    peak_page_util = [0 for _ in range(workers)]

    worker = list(warp(tid, vline, vassoc, vsets, page_size, pages, ipc) for tid in range(workers))

    with open (ctxt_fname, newline='') as rdf:
        reader = csv.reader(rdf, quoting=csv.QUOTE_NONE)
        header = next(reader)
        ctxt = next(reader)
        try:
            wid = next(reader)
        except:
            print (f'{workers_test.__name__} ||| No worker IDs provided!\n\n')
            wid = None
        # Call Scheduler
        scheduler(ctxt, wid, worker, disable=0)
    
    # Get Core stats
    for wid, w in enumerate(worker):
        total_access[wid], ext_hits[wid], int_hits[wid], load_dist[wid], avg_page_util[wid], peak_page_util[wid] = w.print_stats()
        # Reset warp
        w.reset_stats()
        w.reset_bandit()
    
    # Accum stats
    sext_hits = sum(ext_hits)
    l1_hits = sum(int_hits)
    l2_hits = worker[0].get_L2_stats()
    sint_hits = l1_hits + l2_hits
    stot_access = sext_hits + sint_hits

    # Print stats
    hit_rate = sint_hits/stot_access
    miss_rate = sext_hits/stot_access
    avg_pg_util = sum(avg_page_util)/workers
    peak_pg_util = max(peak_page_util)
    print (f'\n{workers_test.__name__} ||| ------ CUMMULATIVE STATS BEGIN ------ ')
    print (f'{workers_test.__name__} ||| Total accesses = {stot_access} | mem_ext_hits = {sext_hits}, mem_int_hits = {sint_hits} | L1 hits = {l1_hits}, L2 hits = {l2_hits} | hit_rate = {hit_rate}, miss_rate = {miss_rate}')
    print (f'{workers_test.__name__} ||| Average Page Utilization = {avg_pg_util}, Peak Average Page Utilization = {peak_pg_util} | Cycles = {load_dist}')
    print (f'{workers_test.__name__} ||| ---------- END ---------- ')

    # Remove all warps
    worker[0].reset_common()

"""
Main function
"""
def main():

    # CacheBandit Mutables
    """
    Set vassoc and vsets to shmem_assoc, shmem_sets to elicit normal (baseline) cache behaviour.
    This removes the Global replacement policy out of the picture, as no overflow happens and deallocate is never called/used, if called.
    """
    vassoc = shmem_assoc        # Use the same number of comparators as Baseline
    vsets = 512                 # Assuming 4 BRAMs are used in parallel 4-way assoc, then each BRAM has a depth of 512 (in 64b mode)
    vsize = vassoc * vsets

    # Run control parameters
    version = 'v2.0.1'
    #ctxt_type = 'streamA'         # Valid types -> streamA, interleavedA
    ctxt_type = 'interleavedA'
    ctxt_path = '.\\context\\'
    log_path = '.\\logs\\'
    run_name = 'cb_test'
    log_run_name = run_name + '_' + str(workers) + '_' + str(contexts) + '_' + version
    #log_run_name = 'dummy' + '_' + str(workers) + '_' + str(contexts) + '_' + version

    # Matrix parameters
    """
    Batch run
    """
    #mat_name = ['bcsstk10', 'bcsstk13', 'bcsstk17', 'c8_mat11', 'cq9', 'fv1', 'kl02', 'lhr34c', 'pdb1HYS', 'psmigr_1', 'wiki-Vote', 'ca-HepTh', 'p2p-Gnutella04', 'as-735', 'amazon0312', 'KM_2000_d100', 'KM_3000_d100', 'KM_4000_d100', 'KM_5000_d100']
    #dyn_sets = [shmem_sets, vsets, shmem_sets, vsets, vsets, shmem_sets, vsets, shmem_sets, shmem_sets, vsets, shmem_sets, shmem_sets, shmem_sets, shmem_sets, shmem_sets, vsets, vsets, vsets, vsets]
    mat_name = ['bcsstk10', 'bcsstk13', 'bcsstk17', 'c8_mat11', 'cq9', 'fv1', 'kl02', 'lhr34c', 'pdb1HYS', 'psmigr_1', 'wiki-Vote', 'ca-HepTh', 'p2p-Gnutella04', 'as-735', 'amazon0312']
    #dyn_sets = [shmem_sets, vsets, shmem_sets, vsets, vsets, shmem_sets, vsets, shmem_sets, shmem_sets, vsets, shmem_sets, shmem_sets, shmem_sets, shmem_sets, shmem_sets]
    dyn_sets = [vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets, vsets]
    """
    Individual run
    """
    #mat_name = ['bcsstk17']
    #dyn_sets = [vsets]

    # Check if Page_size is greater than 4 * IPC (twice the reclaim size) -> Fast to slow CDC check
    if (page_size <= 4 * ipc):
        print (f'{main.__name__} ||| Page size [{page_size}] must be greater than 4*IPC [{4*ipc}]')
        return -1
    # Check if Pages is greater than 2 -> Slow to fast CDC check
    if (pages < 2):
        print (f'{main.__name__} ||| Pages [{pages}] must be greater than 2')
        return -1

    for mid, mname in enumerate(mat_name):
        strmA_fname = ctxt_path + ctxt_type + '_' + run_name + '_' + mname + '.csv'
        log_name = log_path + log_run_name + '_' + mname + '_log.txt'

        matA_header = []
        with open (strmA_fname, newline='') as rdf:
            reader = csv.reader(rdf, quoting=csv.QUOTE_NONE)
            matA_header = next(reader)

        print (f'{main.__name__} ||| Streaming: Matrix A dim K = {matA_header[0]}, M = {matA_header[1]}, NNZ = {matA_header[2]}, Density = {matA_header[3]}, Unique frames visited = {matA_header[4]}')

        with open(log_name, 'w') as f:
            with redirect_stdout(f):
                # Print config
                timestamp = datetime.datetime.now()
                print (f'{main.__name__} ||| ------ CONFIG [{timestamp}] ------')
                print (f'{main.__name__} ||| virtual_line = {shmem_line}, virtual_assoc = {vassoc}, virtual_sets = {vsets} | virtual_size = {vsize} ; {vsize * bytes_per_line} bytes | SharedMemory_Size = {shmem_size}, Pages = {pages}, Page_size = {page_size}')
                print (f'{main.__name__} ||| Matrix A dim K = {matA_header[0]}, M = {matA_header[1]}, NNZ = {matA_header[2]}, Density = {matA_header[3]}, Unique frames visited = {matA_header[4]}')
                print (f'{main.__name__} ||| PE_freq = {freq_PE}, CacheBandit_freq = {freq_cacheBandit}, IPC = {ipc} | Stream type = {ctxt_type}\n')

                workers_test(strmA_fname, shmem_line, shmem_assoc, dyn_sets[mid], page_size, pages, ipc)
                print (f'\n{main.__name__} ||| -------------------------------------- Baseline Run --------------------------------------\n', flush=True)
                workers_test(strmA_fname, shmem_line, shmem_assoc, shmem_sets, page_size, pages, ipc)

if __name__ == "__main__":
    main()