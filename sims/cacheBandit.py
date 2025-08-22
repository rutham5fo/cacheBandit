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


# Consts
life_MAX    = 9999

# Physical memory
# Search Space (BLOCK_D) = 128 lines/frames | 128B per line (8*2 BRAMs) => 16KiB => 4KiB per core (8 threads)
shmem_line = 16
shmem_assoc = 4                            # Associativity per set of cache
shmem_sets = 32                             # No. of sets in cache
shmem_size = shmem_assoc * shmem_sets
page_size = 32                              # !! Page_size must be bigger than IPC (reclaim size) count !!
pages = int(shmem_size/page_size)

# Worker and Thread related params
worker_count = 1
threads = 4                 # No. of thread contexts TDM'd by worker @ Freq_cacheBandit

# CacheBandit constants
freq_cacheBandit = 100 #MHz
freq_PE = threads * freq_cacheBandit #MHz
ipc = int(math.ceil(freq_PE / freq_cacheBandit))
bytes_per_line = 128

# CacheBandit Mutables
"""
Set vassoc and vsets to shmem_assoc, shmem_sets to elicit normal (baseline) cache behaviour.
This removes the Global replacement policy out of the picture, as no overflow happens and deallocate is never called/used, if called.
"""
vassoc = 2
vsets = 512
vsize = vassoc * vsets

# Stats related
ext_mem_access_penalty = 8                  # External memory latency


class bandit:

    def __init__ (self, block_id, virtual_line, virtual_assoc, virtual_sets, page_size, pages, ipc=1):
        self.bid = block_id
        self.vline = virtual_line
        self.vassoc = virtual_assoc
        self.vsets = virtual_sets
        self.vlru = [int(math.ceil(x%self.vassoc)) for x in range(self.vsets)]
        self.vcache = [[[None, (None, None), 0, 0] for _ in range(self.vassoc)] for _ in range(self.vsets)]  # [Key, (phy_ptr, page), lifetime, valid] per assoc
        # CacheBandit
        self.ipc = ipc
        self.init_page = 1
        self.init_invalidate = 1
        # Page
        self.pages = pages
        self.page_size = page_size       # Total number of lines in a page
        self.reclaim_size = self.ipc
        self.refresh_threshold = self.page_size - self.reclaim_size + 1
        self.page_life = [0 for _ in range(self.pages)]
        self.page_util = [0 for _ in range(self.pages)]
        self.cur_sol = [0 for _ in range(self.pages)]
        self.cur_page = 0
        self.next_page = self.cur_page + 1
        self.vmap = [[v for v in range(self.page_size)] for _ in range(self.pages)]
        self.pmap = [[p for p in range(self.page_size)] for _ in range(self.pages)]
        self.smap = [[[0, 0] for _ in range(self.page_size)] for _ in range(self.pages)]
        self.rpmap = [p for p in range(self.page_size)]
        self.rsmap = [[0, 0] for _ in range(self.page_size)]
        self.commit_base_ptr = 0
        # Stats
        self.ext_hit = 0
        self.int_hit = 0
        self.alloc_cnt = 0
        self.dealloc_cnt = 0
        self.accum_cnt = 0
        self.accum_page_util = 0
        self.avg_page_util = 0
        self.peak_page_util = 0
        self.cycle = 0

    def print_stats (self):
        # Compute stats
        total_access = self.ext_hit + self.int_hit
        hit_rate = self.int_hit / total_access
        miss_rate = self.ext_hit / total_access
        self.avg_page_util = self.accum_page_util / self.accum_cnt
        peak_page_util = 1.0 if (self.vassoc == shmem_assoc and self.vsets == shmem_sets) else self.peak_page_util / self.page_size
        print (f'{bandit.print_stats.__name__} ||| ------ BANDIT[{self.bid}] STATS BEGIN ------ ')
        print (f'{bandit.print_stats.__name__} ||| Total accesses = {total_access} | mem_ext_hit = {self.ext_hit}, mem_int_hit = {self.int_hit} | hit_rate = {hit_rate}, miss_rate = {miss_rate}')
        print (f'{bandit.print_stats.__name__} ||| Allocs_called = {self.alloc_cnt}, Deallocs_called = {self.dealloc_cnt} | Average Page Utilization = {self.avg_page_util}, Peak Page Utilization = {peak_page_util}')
        print (f'{bandit.print_stats.__name__} ||| ---------- BANDIT END ---------- ')

        return total_access, self.ext_hit, self.int_hit, self.avg_page_util, peak_page_util, self.cycle

    def reset_stats (self):
        self.ext_hit = 0
        self.int_hit = 0
        self.alloc_cnt = 0
        self.dealloc_cnt = 0
        self.accum_cnt = 0
        self.accum_page_util = 0
        self.avg_page_util = 0
        self.peak_page_util = 0
        self.cycle = 0

    def reset_page (self):
        self.init_page = 1
        self.init_invalidate = 1
        self.page_util = [0 for _ in range(self.pages)]
        self.cur_sol = [0 for _ in range(self.pages)]
        self.cur_page = 0
        self.seed_rdy = 0
        self.next_page = self.cur_page + 1
        self.vmap = [[v for v in range(self.page_size)] for _ in range(self.pages)]
        self.pmap = [[p for p in range(self.page_size)] for _ in range(self.pages)]
        self.smap = [[[0, 0] for _ in range(self.page_size)] for _ in range(self.pages)]
        self.rpmap = [p for p in range(self.page_size)]
        self.rsmap = [[0, 0] for _ in range(self.page_size)]
        self.commit_base_ptr = 0

    def get_alloc_cnt (self):
        return self.alloc_cnt
    
    def get_dealloc_cnt (self):
        return self.dealloc_cnt
    
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
    
    def allocate (self, set_sel, line_id):
        # Check for page overflow
        if (self.cur_sol[self.cur_page] == self.page_size):
            print (f'{bandit.allocate.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Page-Overflow ! | cur_sol = {self.cur_sol[self.cur_page]}, page_size = {self.page_size}')
            raise IndexError("Page-Overflow!")
        sol_sel = self.cur_sol[self.cur_page]
        prev_sol = self.pmap[self.cur_page][sol_sel]
        prev_set = self.smap[self.cur_page][sol_sel][0]
        prev_line = self.smap[self.cur_page][sol_sel][1]
        self.smap[self.cur_page][sol_sel][0] = set_sel
        self.smap[self.cur_page][sol_sel][1] = line_id
        self.cur_sol[self.cur_page] += 1
        #print (f'{bandit.allocate.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Allocating line {prev_sol} | new_sol = {self.cur_sol} | set_sel = {set_sel}, line_id = {line_id}', flush=True)
        self.page_util[self.cur_page] += 1
        self.alloc_cnt += 1
        self.peak_page_util = self.page_util[self.cur_page] if (self.page_util[self.cur_page] > self.peak_page_util) else self.peak_page_util
        
        # Housekeeping
        self.clean(self.page_util[self.cur_page])

        return (prev_sol, self.cur_page, prev_set, prev_line)
    
    def dealloc_refresh (self, page):
        # Fall back to FIFO reclaim policy if seed is none
        beg = self.ipc - 1
        end = self.page_size - 1
        vptr = random.randint(beg, end)
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
                print (f'{bandit.dealloc_refresh.__name__} [{self.bid}] | Page[{this_page}] ||| {e} | base_ptr = {base_ptr}, vptr = {vptr}, reclaim_size = {self.reclaim_size}', flush=True)
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
        self.next_page = this_page
        #print (f'{bandit.dealloc_refresh.__name__}[{self.bid}] |  next_page = {self.next_page}')

    def dealloc_commit (self, page):
        # Merge rmap with pmap
        delim = self.commit_base_ptr - self.reclaim_size + 1
        #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| commit_ptr = {self.commit_base_ptr}, cur_sol = {self.cur_sol[page]}, pmap = {self.pmap[page]}', flush=True)
        if (delim != 0):
            self.rpmap[0:delim] = self.pmap[page][0:delim]
            self.rsmap[0:delim] = self.smap[page][0:delim]
        #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| Post-shift', flush=True)
        #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| pmap = {self.rpmap}', flush=True)
        self.cur_sol[page] -= self.reclaim_size
        self.page_util[page] -= self.reclaim_size
        self.dealloc_cnt += 1
        # Refresh page
        self.pmap[page] = self.rpmap
        self.smap[page] = self.rsmap
        #print (f'{bandit.dealloc_commit.__name__}[{self.bid}] | Page[{page}] ||| cur_sol = {self.cur_sol[page]}, next_alloc = {self.pmap[page][self.cur_sol[page]]}', flush=True)
    
    def clean (self, page_util):
        this_page_util = page_util
        # Get LRU page
        page = self.page_lru()
        # Check page utilization
        #print (f'{bandit.clean.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Current Page utilization = {this_page_util}, dealloc_refresh_threshold = {self.refresh_threshold} | init_page = {self.init_page} | Page_lru = {page}', flush=True)
        if ((this_page_util == self.refresh_threshold and not self.init_page) or (this_page_util == self.refresh_threshold and self.init_page and self.cur_page == self.pages-1)):
            self.dealloc_refresh(page)
        elif ((this_page_util == self.page_size and not self.init_page) or (this_page_util == self.page_size and self.init_page and self.cur_page == self.pages-1)):
            self.dealloc_commit(self.next_page)
            #print (f'{bandit.clean.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Page switch. Next page = {self.next_page}')
            self.cur_page = self.next_page
            self.init_page = 0
        elif (this_page_util == self.page_size and self.init_page):
            #print (f'{bandit.clean.__name__}[{self.bid}] | Page[{self.cur_page}] ||| Page switch. Next page = {self.cur_page+1}')
            self.cur_page += 1

    def invalidate (self, alloc, set_sel, line):
        if (alloc and not self.init_invalidate):
                #print (f'{bandit.invalidate.__name__} ||| Invalidating set[{set_sel}], line[{line}] = {self.vcache[set_sel][line]}', flush=True)
                self.vcache[set_sel][line][3] = 0
        elif (self.init_invalidate and not self.init_page):
            self.init_invalidate = 0

    def find_lru (self, set_sel, vset):
        base_life = life_MAX
        comparator = []
        for line_id, line in enumerate(vset):
            line_life = line[2]
            comparator.append([line_life, line_id])
            #print (f'{bandit.find_lru.__name__} [{self.bid}] ||| LRU || Frame = {line[0]}, phy_ptr = {line[1]}, line_life = {line_life}, base_life = {base_life}', flush=True)
        lru_meta = min(comparator)
        lru_id = lru_meta[1]
        self.vlru[set_sel] = lru_id
        #print (f'{bandit.find_lru.__name__} [{self.bid}] ||| LRU || list = {self.vlru}')
    
    def age_cache(self):
        for sid, s in enumerate(self.vcache):
            for lid, line in enumerate(s):
                if (line[2] > 0):
                    line[2] -= 1
                    #if (line[2] < 0):
                    #    print (f'{bandit.age_cache.__name__} ||| Age turns negative!')
                    #    sys.exit(1)
    
    def show_cache(self, set_sel):
        print (f'{bandit.show_cache.__name__} [{self.bid}] ||| Dumping Cache state:\n')
        for set_id, s in enumerate(self.vcache):
            if (set_id == set_sel or set_sel == self.vsets):
                for line_id, line in enumerate(s):
                    print (f'{bandit.show_cache.__name__} [{self.bid}] ||| Set[{set_id}] || Line[{line_id}] = {line}', flush=True)
    
    def find_L1 (self, set_sel, key):
        found = 0
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
        for line_id, line in enumerate(s):
            if (line[0] == key and line[3]):
                #print (f'{bandit.find_L1.__name__} [{self.bid}] ||| Found key [{key}] in L1 at location [{set_sel}][{line_id}] | line = {line}', flush=True)
                found = 1
                at = line_id
                ptr = line[1][0]
                page = line[1][1]
                life = life_MAX
                vld = line[3]
        
        #print (f'{bandit.find_L1.__name__} [{self.bid}] ||| found = {found}, at = {at}, ptr = {ptr}, page = {page}, life = {life}, vld = {vld} | alloc = {alloc}', flush=True)

        if (found == 0):
            if (vld):
                alloc = 0
                ptr, page, invld_set, invld_line = ptr, page, invld_set, invld_line
            else:
                alloc = 1
                ptr, page, invld_set, invld_line = self.allocate(set_sel, at)
            life = life_MAX
            vld = 1
            # Evict current LRU line, and insert new line
            #print (f'{bandit.find_L1.__name__} [{self.bid}] ||| Inserting key [{key}] in L1 at location [{set_sel}][{ev_lid}] | evicted_line = {evict_line}, inserted_line = {[key, (ptr, page), life, vld]} | alloc = {alloc}', flush=True)
            
        insert_line = [key, (ptr, page), life, vld]

        # Insert line
        s[at] = insert_line

        # Invalidate line
        self.invalidate(alloc, invld_set, invld_line)

        # Update LRU of current set
        self.find_lru(set_sel, s)

        self.page_touch(page)

        return found
        
    def strm (self, taddr):
        found_L1 = 0
        # Age cache
        self.age_cache()
        # Age page
        self.age_page()
        # Read address from context stream
        addr = int(taddr)
        vframe = int(math.floor(addr / self.vline))
        vset_sel = int(math.floor(vframe % self.vsets))
        #print (f'{bandit.strm.__name__} ||| ------- RUN Bandit [{self.bid}] ------- || strm_addr = {taddr}, hit_level = {hit_level} | Key = {vframe} | Set = {vset_sel}', flush=True)
        
        # Find addr in Cache
        found_L1 = self.find_L1(vset_sel, vframe)
        
        # Increament Penalty/load cost + Stats
        if (found_L1):
            self.int_hit += 1
            self.cycle += 1
        else:
            self.ext_hit += 1
            self.cycle += ext_mem_access_penalty
        # Accumulate Page Utilization
        norm_page_util = self.page_util[self.cur_page] / self.page_size
        self.accum_page_util += norm_page_util
        self.accum_cnt += 1

def scheduler (ctxt, worker):
    wid = 0
    for elem in ctxt:
        taddr = int(elem)
        worker[wid].strm(taddr)

def workers_test (ctxt_fname, vline, vassoc, vsets, page_size, pages, ipc):
    total_access = [0 for _ in range(worker_count)]
    int_hits = [0 for _ in range(worker_count)]
    ext_hits = [0 for _ in range(worker_count)]
    avg_page_util = [0 for _ in range(worker_count)]
    peak_page_util = [0 for _ in range(worker_count)]
    load_dist = [0 for _ in range(worker_count)]
    worker = list(bandit(bid, vline, vassoc, vsets, page_size, pages, ipc) for bid in range(worker_count))
    
    with open (ctxt_fname, newline='') as rdf:
        reader = csv.reader(rdf, quoting=csv.QUOTE_NONE)
        header = next(reader)
        ctxt = next(reader)
        # Call Scheduler
        scheduler(ctxt, worker)
    
    # Get stats
    for wid, w in enumerate(worker):
        total_access[wid], ext_hits[wid], int_hits[wid], avg_page_util[wid], peak_page_util[wid], load_dist[wid] = w.print_stats()
        w.reset_stats()
        w.reset_page()

    # Accum stats
    sext_hits = sum(ext_hits)
    l1_hits = sum(int_hits)
    savg_page_util = sum(avg_page_util)
    speak_page_util = sum(peak_page_util)
    sint_hits = l1_hits
    stot_access = sext_hits + sint_hits

    # Print stats
    hit_rate = sint_hits/stot_access
    miss_rate = sext_hits/stot_access
    avg_util = savg_page_util/worker_count
    peak_avg_util = speak_page_util/worker_count
    print (f'\n{workers_test.__name__} ||| ------ CUMMULATIVE STATS BEGIN ------ ')
    print (f'{workers_test.__name__} ||| Total accesses = {stot_access} | mem_ext_hits = {sext_hits}, mem_int_hits = {sint_hits} | L1 hits = {l1_hits} | hit_rate = {hit_rate}, miss_rate = {miss_rate}')
    print (f'{workers_test.__name__} ||| Average Page Utilization = {avg_util}, Peak Average Page Utilization = {peak_avg_util} | Cycles = {load_dist}')
    print (f'{workers_test.__name__} ||| ---------- END ---------- ')

def compare_test (ctxt_fname, vline, vassoc, vsets, page_size, pages, ipc):
    workers_test(ctxt_fname, vline, vassoc, vsets, page_size, pages, ipc)
    print (f'\n{compare_test.__name__} ||| -------------------------------------- Baseline Run --------------------------------------\n', flush=True)
    workers_test(ctxt_fname, shmem_line, shmem_assoc, shmem_sets, page_size, pages, ipc)

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
    ctxt_path = '.\\context\\'
    log_path = '.\\logs\\'
    run_name = 'cb_test'
    log_run_name = run_name + '_' + str(worker_count) + '_' + str(threads)
    #log_run_name = 'dummy' + '_' + str(worker_count) + '_' + str(threads)

    # Matrix parameters
    mat_name = ['bcsstk10', 'bcsstk13', 'bcsstk17', 'c8_mat11', 'cq9', 'fv1', 'kl02', 'lhr34c', 'pdb1HYS', 'psmigr_1', 'wiki-Vote', 'ca-HepTh', 'p2p-Gnutella04', 'as-735', 'amazon0312', 'KM_2000_d100', 'KM_3000_d100', 'KM_4000_d100', 'KM_5000_d100']
    dyn_sets = [shmem_sets, vsets, shmem_sets, vsets, vsets, shmem_sets, vsets, shmem_sets, shmem_sets, vsets, shmem_sets, shmem_sets, shmem_sets, shmem_sets, shmem_sets, vsets, vsets, vsets, vsets]
    #mat_name = ['bcsstk17']
    #dyn_sets = [44]

    # Check if Page_size is greater than IPC
    if (page_size <= ipc):
        print (f'{main.__name__} ||| Page size [{page_size}] must be greater than IPC [{ipc}]')
        return -1

    for mid, mname in enumerate(mat_name):
        strmA_fname = ctxt_path + 'streamA_' + run_name + '_' + mname + '.csv'
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
                print (f'{main.__name__} ||| PE_freq = {freq_PE}, CacheBandit_freq = {freq_cacheBandit}, IPC = {ipc}\n')

                #workers_test(strmA_fname, shmem_line, vassoc, dyn_sets[mid], page_size, pages, ipc)
                compare_test(strmA_fname, shmem_line, vassoc, dyn_sets[mid], page_size, pages, ipc)

if __name__ == "__main__":
    main()