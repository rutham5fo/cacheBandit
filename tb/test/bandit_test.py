import math
import os
import sys
import time
import random
import logging
import csv
from config.pucb_intf_cfg import config, default_params
from cosim.bandit import bandit

class warp():
    # Each warp has 'context' number of threads
    def __init__(self, warp_name: str, warp_id: int, bandit_config: config, external_access_latency: int, logger_name: str = ""):
        # Get logger
        self.logger = logging.getLogger(logger_name)
        # Initialize bandit
        self.name = warp_name
        self.wid = warp_id
        self.cb_cfg = bandit_config
        self.cb = bandit(self.name, self.cb_cfg, no_score=0, logger_name=logger_name)
        self.threads = self.cb.bsize
        self.mode = 0           # Mode can take the following vals: 0 = independant (default), 1 = psuedo LLC en, 2 = only field view shared, 3 = All shared, 4 = All shared and set limiter enabled
        self.external_access_latency = external_access_latency
        # Stats
        self.total_access = 0
        self.ignored = 0
        self.internal = 0
        self.external = 0

    def reset(self):
        self.cb.run(reset=1)
        self.mode = 0
        # Stats
        self.total_access = 0
        self.ignored = 0
        self.internal = 0
        self.external = 0

    def get_stats(self):
        ret_val = None

        # Compute time spent by warp performing control/memory access (excludes compute)
        external_memory_time = self.external*self.external_access_latency
        search_memory_time = self.ignored
        total_memory_time = external_memory_time + search_memory_time + self.internal
        # Calculate the ratio of psuedo LLC mode search latency and external access latency
        pLLC_extern_ratio = self.external_access_latency/self.threads
        # Other stats
        total_access = self.total_access
        external_access = self.external
        hit_rate = self.internal/self.total_access if (self.total_access != 0) else 0
        miss_rate = 1.0 - hit_rate
        
        ret_val = (total_memory_time, external_memory_time, search_memory_time, pLLC_extern_ratio, total_access, external_access, hit_rate, miss_rate)
        
        return ret_val
    
    def set_external_access_latency(self, external_access_latency):
        self.external_access_latency = external_access_latency

    def set_mode(self, mode=0):
        bandit_mode_sel = 0 if (mode == 0 or mode == 1) else 1     # Bandit can be in two modes: 0 = Private (view private table), 1 = Shared (view field table)
        page_share_en = 1 if (mode == 3 or mode == 4) else 0
        limit_en = 1 if (mode == 4) else 0
        self.logger.info(f'{warp.set_mode.__name__} ||| Changing warp {self.name}\'s mode from {self.mode} to: {mode} | bandit_mode_sel = {bandit_mode_sel}, page_share_en = {page_share_en}, limit_en = {limit_en}')
        for tid in range(self.threads):
            # Ignore miss needs to be set, inorder to route channel_sel through cb's internal tdm_sel
            self.cb.run(mode_sel=bandit_mode_sel, channel_sel=tid, ignore_miss=1, mode_switch_en=1, page_share_en=page_share_en, limit_en=limit_en)
        self.mode = mode

    def llc_run(self, thread, addr, wen=0):
        # Only the last miss is considered
        cb_miss = 0
        cb_hit = 0
        cid = int(math.ceil((thread+1)%self.threads))
        for tid in range(self.threads):
            # Check all channels till found
            # Last miss (which will not be ignored) always lands on the correct channel
            miss_ignore = 1 if (tid != self.threads-1) else 0
            cb_out = self.cb.run(channel_sel=cid, addr=addr, wen=wen, bandit_en=1, ignore_miss=miss_ignore)
            cb_miss, cb_hit = cb_out[1], cb_out[2]
            if (cb_hit):
                self.logger.debug(f'{warp.llc_run.__name__} ||| Hit for addr[{addr}] in thread[{cid}]')
                break
            else:
                self.ignored += 1
                self.logger.debug(f'{warp.llc_run.__name__} ||| Miss for addr[{addr}] in thread[{cid}]')
            cid = cid+1 if (cid != self.threads-1) else 0
        return cb_miss
    
    def norm_run(self, thread, addr, wen=0):
        # Single run, miss considered
        # Force cb_tdm_id to thread
        self.cb.tdm_id = thread
        # Run bandit
        cb_out = self.cb.run(addr=addr, wen=wen, bandit_en=1)
        cb_miss = cb_out[1]
        if (cb_miss):
            self.logger.debug(f'{warp.norm_run.__name__} ||| Miss for addr[{addr}] in thread[{thread}]')
        return cb_miss
    
    def dma(self):
        self.external += 1

    def memory(self, direct_en=0):
        if (direct_en):
            # TODO
            pass
        else:
            self.internal += 1
            
    def exec(self, thread, addr, wen=0, direct_en=0):
        cb_miss = 0
        # Run Bandit
        if (direct_en):
            # TODO
            pass
        else:
            self.total_access += 1
            if (self.mode == 1):
                cb_miss = self.llc_run(thread, addr, wen)
            else:
                cb_miss = self.norm_run(thread, addr, wen)
        # Run DMA or internal memory
        if (cb_miss):
            self.dma()
        else:
            self.memory(direct_en)


class warp_stats():

    def __init__(self, name, workers, disabled=0, logger_name=""):
        self.logger = logging.getLogger(logger_name)
        self.name = name
        self.workers = workers
        self.disabled = disabled
        ## Basic stats
        self.tmt = [0 for _ in range(self.workers)]                      # Total time spent in memory operations
        self.emt = [0 for _ in range(self.workers)]                      # Time spent with external memory access
        self.smt = [0 for _ in range(self.workers)]                      # Time spent with searching internal memory
        self.llc_extern_ratio = [0 for _ in range(self.workers)]         # Ratio of threads searched to the external access latency
        self.taccess = [0 for _ in range(self.workers)]                  # Total access registered
        self.eaccess = [0 for _ in range(self.workers)]                  # External access registered
        self.hit_rate = [0 for _ in range(self.workers)]                 # Hit rate
        self.miss_rate = [0 for _ in range(self.workers)]                # Miss rate
        ## Accum stats
        self.acc_tmt = 0
        self.acc_emt = 0
        self.acc_smt = 0
        self.acc_llcext_ratio = 0
        self.acc_taccess = 0
        self.acc_eaccess = 0
        self.acc_hit_rate = 0
        self.acc_miss_rate = 0

    def accum(self):
        ## Accum stats
        self.acc_tmt = sum(self.tmt)
        self.acc_emt = sum(self.emt)
        self.acc_smt = sum(self.smt)
        self.acc_llcext_ratio = sum(self.llc_extern_ratio)/self.workers
        self.acc_taccess = sum(self.taccess)
        self.acc_eaccess = sum(self.eaccess)
        self.acc_iaccess = self.acc_taccess - self.acc_eaccess
        self.acc_hit_rate = sum(self.hit_rate)
        self.acc_miss_rate = sum(self.miss_rate)
    
    def log_accum(self):
        # CB Logs
        self.logger.info(f'{warp_stats.log_accum.__name__} ||| ------ CUMMULATIVE "{self.name}" STATS BEGIN ------ ')
        self.logger.info(f'{warp_stats.log_accum.__name__} ||| Total accesses = {self.acc_taccess}, Hits = {self.acc_iaccess}, Miss = {self.acc_eaccess}')
        self.logger.info(f'{warp_stats.log_accum.__name__} ||| Hit_rate = {self.acc_hit_rate}, Miss_rate = {self.acc_miss_rate}, Virtual LLC search to External access ratio = {self.acc_llcext_ratio}')
        self.logger.info(f'{warp_stats.log_accum.__name__} ||| Total cycles on memory access = {self.acc_tmt}, Cycles spent on DRAM = {self.acc_emt}, Cycles spent on Searching = {self.acc_smt}')
        self.logger.info(f'{warp_stats.log_accum.__name__} ||| ---------- END ---------- \n')
    
    def get_reduction(self, sub_stats, round_pts=0):
        ret_val = None
        if (isinstance(sub_stats, warp_stats) and sub_stats.disabled != 1 and self.disabled != 1):
            # Compute DRAM reduction with self as reference
            eaccess_red = (sub_stats.acc_eaccess-self.acc_eaccess)/sub_stats.acc_eaccess if (sub_stats.acc_eaccess != 0) else 0
            tmt_red = (sub_stats.acc_tmt-self.acc_tmt)/sub_stats.acc_tmt if (sub_stats.acc_tmt != 0) else 0
            eaccess_red_rnd = round(eaccess_red, round_pts) if (round_pts != 0) else eaccess_red
            tmt_red_rnd = round(tmt_red, round_pts) if (round_pts != 0) else tmt_red
            ret_val = (eaccess_red_rnd, tmt_red_rnd)
            self.logger.debug(f'{warp_stats.get_reduction.__name__} ||| Reduction of "{self.name}" with respect to "{sub_stats.name}": DRAM = {ret_val[0]}; TIME = {ret_val[1]}\n\n')
        return ret_val

"""
---------------------- BEGIN ROUTINES --------------------------
"""

def step_test(cb, page_size, assoc, sets, shared=1, logger_name=""):
    logger = logging.getLogger(logger_name)

    # Bandit params
    bid = 0
    assoc = 4
    sets = 8
    page_size = 8
    pages = 2
    shared = 0

    # Instantiate bandit
    def_params = default_params()
    cfg_params = def_params.get_params()
    cfg_params['WORDS_PER_LINE'] = 1
    cfg_params['ASSOC'] = assoc
    cfg_params['SETS'] = sets
    cfg_params['PAGES'] = pages
    cfg_params['PAGE_SIZE'] = page_size
    cfg_params['PIPE_DEP'] = 2
    cfg = config(cfg_params)
    cb = bandit(bid, cfg, no_score=1, logger_name=logger_name)

    phy_sets = int(page_size/assoc)
    logger.info(f'\n{step_test.__name__} || Starting Step test: phy_sets = {phy_sets}, field_sets = {sets}, field_assoc = {assoc}')

    cb_en = 1
    exit = ''
    while exit != 'q':
        addr = int(input('Enter an address: '))
        wen = int(input('Enter wen: '))
        print(f'{step_test.__name__} || Sending addr = {addr}, wen = {wen} to bandit')
        logger.info(f'{step_test.__name__} || Sending addr = {addr}, wen = {wen} to bandit')
        cb_consume, cb_miss, mem_addr, mem_wen, dma_rd_en, dma_rd_addr, heap_wr_addr, dma_wr_en, dma_wr_addr, heap_rd_addr = cb.run(addr=addr, wen=wen, cb_en=cb_en, shared=shared)
        logger.info(f'{step_test.__name__} || cb_consume = {cb_consume}, cb_miss = {cb_miss}, mem_addr = {mem_addr}, mem_wen = {mem_wen}')
        logger.info(f'{step_test.__name__} || dma_rd_addr = {dma_rd_addr}, heap_wr_addr = {heap_wr_addr}, dma_rd_en = {dma_rd_en}, dma_wr_addr = {dma_wr_addr}, heap_rd_addr = {heap_rd_addr}, dma_wr_en = {dma_wr_en}\n')
        print(f'{step_test.__name__} || cb_consume = {cb_consume}, cb_miss = {cb_miss}, mem_addr = {mem_addr}, mem_wen = {mem_wen}')
        print(f'{step_test.__name__} || dma_rd_addr = {dma_rd_addr}, heap_wr_addr = {heap_wr_addr}, dma_rd_en = {dma_rd_en}, dma_wr_addr = {dma_wr_addr}, heap_rd_addr = {heap_rd_addr}, dma_wr_en = {dma_wr_en}')
        exit = input('Press \'q\' to quit: ')

def hamming_dist(dist_var, assoc):
    dist = 0
    for _ in range(assoc):
        lsb = dist_var & 0x1
        dist += lsb
        dist_var = dist_var >> 1
    return dist

def lru_test(run_len, assoc, cb, logger_name=""):
    logger = logging.getLogger(logger_name)

    # Generate run_len numbers of stride assoc, to cause
    # set thrashing in set 0. Assert hamming distance is
    # 1 between generated pLRU lines and hamming distance
    # of 1 with the previous state, i.e., make sure every 
    # consecutive 'assoc' miss generates corresponding unique vlaues.

    logger.info(f'{lru_test.__name__} ||| Starting pLRU test for assoc = {assoc}')

    set_sel = 0
    prev_state = 0
    cur_state = 0
    for _ in range(run_len):
        flru_line = cb.get_lru(set_sel, 0)
        state_in = 1 << flru_line
        next_state = cur_state ^ state_in
        cn_val = cur_state ^ next_state
        pn_val = prev_state ^ next_state
        logger.debug(f'{lru_test.__name__} ||| prev_state = {prev_state}, cur_state = {cur_state}, next_state = {next_state} | flru_line = {flru_line}, cn_val = {cn_val}, pn_val = {pn_val}')
        cn_hd = hamming_dist(cn_val, assoc)
        assert (cn_hd), logger.error(f'{lru_test.__name__} ||| pLRU Test failed: LINE_GEN_OVERFLOW : Generated line lies outside assoc range : prev_state = {prev_state}, cur_state = {cur_state}, next_state = {next_state} | cn_hd = {cn_hd}')
        pn_hd = hamming_dist(pn_val, assoc)
        assert (pn_hd), logger.error(f'{lru_test.__name__} ||| pLRU Test failed: REPEAT_LINE_GEN : Generated line same as previous line : prev_state = {prev_state}, cur_state = {cur_state}, next_state = {next_state} | pn_hd = {pn_hd}')

        cb.put_lru(set_sel, flru_line, 1, 0)
        prev_state = cur_state
        cur_state = next_state
    
    logger.info(f'{lru_test.__name__} ||| pLRU test complete')

def auto_test(run_len=1000, logger_name=""):
    logger = logging.getLogger(logger_name)

    ## Default params
    workers = 4
    words_per_line = 16
    assoc = 4
    sets = 256
    page_size = 128

    logger.info(f'{auto_test.__name__} || Starting Auto test: words_per_line = {words_per_line}, assoc = {assoc}, sets = {sets}, page_size = {page_size}, workers = {workers}\n')

    ## Get params template
    def_params = default_params()
    cfg_params = def_params.get_params()

    ## Instantiate baseline
    base_workers = workers
    baseline_sets = int(page_size/assoc)
    cfg_params['SETS'] = baseline_sets
    cfg_params['PAGES'] = 1
    cfg_params['PAGE_SIZE'] = page_size
    cfg_params['PIPE_DEP'] = 2
    cfg = config(cfg_params)
    logger.info(f'{auto_test.__name__} ||| BASE INIT')
    base = list(bandit(bid, cfg, no_score=1, logger_name=logger_name) for bid in range(base_workers))
    ## Instantiate bandit(s)
    cb_workers = 1
    cb_field_boundaries = int(math.ceil(math.log(baseline_sets, 2)))
    cfg_params['WORDS_PER_LINE'] = words_per_line
    cfg_params['ASSOC'] = assoc
    cfg_params['SETS'] = sets
    cfg_params['PAGES'] = workers
    cfg_params['PAGE_SIZE'] = page_size
    cfg_params['PIPE_DEP'] = 2
    cfg = config(cfg_params)
    logger.info(f'{auto_test.__name__} ||| CB INIT')
    cb = [bandit(0, cfg, no_score=1, logger_name=logger_name)]

    ## Reset all pages
    for w in base:
        w.run(reset=1)
    for w in cb:
        # No page is shared
        w.run(reset=1)

    ## Run LRU test
    lru_test(run_len, assoc, cb[0], logger_name)

    ## Reset cb
    for w in cb:
        # No page is shared
        w.run(reset=1)

    #return

    addr_max = sys.maxsize * 2 + 1
    for r in range(run_len):
        wsel = r
        addr = random.randint(0, addr_max)
        ## Normalize worker select signal
        if (wsel >= workers): wsel = int(math.ceil(wsel%workers))
        ## Run baseline
        base_out = base[wsel].run(addr=addr, cb_en=1)
        ## Run CB
        # CB mem_shared is 0 to mimic baseline
        cb_out = cb[0].run(addr=addr, cb_en=1)
        ## Compare outputs 'consume' and 'miss
        base_val = base_out[0:2]
        cb_val = cb_out[0:2]
        for bout, cout in zip(base_val, cb_val):
            assert (bout == cout), logger.error(f'{auto_test.__name__} ||| Mismatch at: base_vals (consume, miss) = {base_val}, cb_vals = {cb_val} | wsel = {wsel}, addr = {addr}, tdm_channel = {cb[0].tdm_id-1 if (cb[0].tdm_id != 0) else cb[0].bsize-1}')
                
def _test(test_type=0, wsl_en=0):

    log_name = 'bandit_self_test.log'
    log_dir = './cosim/logs/' if (wsl_en == 1) else 'cosim\\logs\\'
    log_path = log_dir + log_name

    logger_name = __name__
    logger = logging.getLogger(logger_name)
    logging.basicConfig(filename=log_path, style="{", filemode='w')
    logger.setLevel(logging.INFO)

    random_seed = time.process_time_ns()
    random.seed(random_seed)
    logger.info(f'{_test.__name__} || PRNG seed = {random_seed}')

    if (test_type == 1):
        auto_test(logger_name=logger_name)
    else:
        step_test(logger_name=logger_name)
    
    logger.info(f'{_test.__name__} ||-------------------------TEST_COMPLETE-------------------------')

def scheduler (ctxt, ctxt_id, shared, field, pLLC, base, full_LRU, workers, contexts, dis_shared=0, dis_field=0, dis_pLLC=0, dis_base=0, dis_full_LRU=0, logger_name=""):
    logger = logging.getLogger(logger_name)
    cid = -1
    logger.info(f'{scheduler.__name__} ||| Starting Scheduler | workers = {workers}, ctxt/thread_ids = {ctxt_id}')
    for id, elem in enumerate(ctxt):
        cid = cid+1 if (ctxt_id is None) else int(ctxt_id[id])
        cb_tdm_id = int(math.ceil(cid%contexts))
        wsel = int(cid/contexts)
        taddr = int(elem)
        ## Normalize worker select signal
        if (wsel >= workers): wsel = int(math.ceil(wsel%workers))
        ## Run modes
        if (dis_base != 1):
            base[wsel].exec(cid, taddr)
        if (dis_pLLC != 1):
            pLLC[wsel].exec(cid, taddr)
        if (dis_field != 1):
            field[wsel].exec(cid, taddr)
        if (dis_shared != 1):
            shared[wsel].exec(cid, taddr)
        if (dis_full_LRU != 1):
            full_LRU[wsel].exec(cid, taddr)
    logger.info(f'{scheduler.__name__} ||| Schedule Complete')

def run_workers(shared, field, pLLC, base, full_LRU, workers, contexts, ctxt_fname, dis_shared=0, dis_field=0, dis_pLLC=0, dis_base=0, dis_full_LRU=0, logger_name=""):
    logger = logging.getLogger(logger_name)

    ## Stats
    base_stats = warp_stats("base", workers, dis_base, logger_name)
    pLLC_stats = warp_stats("pLLC", workers, dis_pLLC, logger_name)
    field_stats = warp_stats("field", workers, dis_field, logger_name)
    shared_stats = warp_stats("shared", workers, dis_shared, logger_name)
    full_LRU_stats = warp_stats("full_LRU", workers, dis_full_LRU, logger_name)

    ## Run workers
    with open (ctxt_fname, newline='') as rdf:
        reader = csv.reader(rdf, quoting=csv.QUOTE_NONE)
        header = next(reader)
        ctxt = next(reader)
        try:
            ctxt_id = next(reader)
        except:
            print (f'{run_workers.__name__} ||| No context IDs provided!\n\n')
            ctxt_id = None
        # Call Scheduler
        scheduler(ctxt, ctxt_id, shared, field, pLLC, base, full_LRU, workers, contexts, dis_shared=dis_shared, dis_field=dis_field, dis_pLLC=dis_pLLC, dis_base=dis_base, dis_full_LRU=dis_full_LRU, logger_name=logger_name)
    
    ## Get stats
    for wid in range(workers):
        base_stats.tmt[wid], base_stats.emt[wid], base_stats.smt[wid], base_stats.llc_extern_ratio[wid], base_stats.taccess[wid], base_stats.eaccess[wid], base_stats.hit_rate[wid], base_stats.miss_rate[wid] = base[wid].get_stats()
        pLLC_stats.tmt[wid], pLLC_stats.emt[wid], pLLC_stats.smt[wid], pLLC_stats.llc_extern_ratio[wid], pLLC_stats.taccess[wid], pLLC_stats.eaccess[wid], pLLC_stats.hit_rate[wid], pLLC_stats.miss_rate[wid] = pLLC[wid].get_stats()
        field_stats.tmt[wid], field_stats.emt[wid], field_stats.smt[wid], field_stats.llc_extern_ratio[wid], field_stats.taccess[wid], field_stats.eaccess[wid], field_stats.hit_rate[wid], field_stats.miss_rate[wid] = field[wid].get_stats()
        shared_stats.tmt[wid], shared_stats.emt[wid], shared_stats.smt[wid], shared_stats.llc_extern_ratio[wid], shared_stats.taccess[wid], shared_stats.eaccess[wid], shared_stats.hit_rate[wid], shared_stats.miss_rate[wid] = shared[wid].get_stats()
        full_LRU_stats.tmt[wid], full_LRU_stats.emt[wid], full_LRU_stats.smt[wid], full_LRU_stats.llc_extern_ratio[wid], full_LRU_stats.taccess[wid], full_LRU_stats.eaccess[wid], full_LRU_stats.hit_rate[wid], full_LRU_stats.miss_rate[wid] = full_LRU[wid].get_stats()
    
    ## Accum stats
    base_stats.accum()
    pLLC_stats.accum()
    field_stats.accum()
    shared_stats.accum()
    full_LRU_stats.accum()

    ## Log Stats
    base_stats.log_accum()
    pLLC_stats.log_accum()
    field_stats.log_accum()
    shared_stats.log_accum()
    full_LRU_stats.log_accum()

    ## Caculate DRAM and Time reduction matrix
    stats_list = [base_stats, pLLC_stats, field_stats, shared_stats, full_LRU_stats]
    reduction_mat = []
    for tar_id, target_stat in enumerate(stats_list):
        reduction_mat.append([])
        for ref_stat in stats_list:
            red_val = target_stat.get_reduction(ref_stat, round_pts=3)
            app_val = (None, None) if (red_val is None) else tuple(red_val)
            reduction_mat[tar_id].append(app_val)

    # Print Reduction matrix
    logger.info(f'{run_workers.__name__} ||| Stats order: {[s.name for s in stats_list]}')
    logger.info(f'{run_workers.__name__} ||| Reduction matrix:')
    _ = [logger.info(f'{run_workers.__name__} ||| {x}') for x in reduction_mat]
    logger.info('\n\n')

def cacheBandit_test(workers, contexts, ctxt_fname, vwords, vassoc, vsets, page_size, external_access_latency=8, dis_shared=0, dis_field=0, dis_pLLC=0, dis_base=0, dis_full_LRU=0, logger_name=''):
    logger = logging.getLogger(logger_name)

    ## Get params template
    def_params = default_params()
    cfg_params = def_params.get_params()

    ## Instantiate bandit(s)
    cb_pages = contexts
    cfg_params['WORDS_PER_LINE'] = vwords
    cfg_params['ASSOC'] = vassoc
    cfg_params['SETS'] = vsets
    cfg_params['PAGES'] = cb_pages
    cfg_params['PAGE_SIZE'] = page_size
    cfg_params['PIPE_DEP'] = 2
    cb_cfg = config(cfg_params)
    logger.info(f'{cacheBandit_test.__name__} ||| WORKER INIT')
    base = list(warp("base", wid, cb_cfg, external_access_latency=external_access_latency, logger_name=logger_name) for wid in range(workers))
    pLLC = list(warp("pLLC", wid, cb_cfg, external_access_latency=external_access_latency, logger_name=logger_name) for wid in range(workers))
    field = list(warp("field", wid, cb_cfg, external_access_latency=external_access_latency, logger_name=logger_name) for wid in range(workers))
    shared = list(warp("shared", wid, cb_cfg, external_access_latency=external_access_latency, logger_name=logger_name) for wid in range(workers))
    full_LRU = list(warp("full_LRU", wid, cb_cfg, external_access_latency=external_access_latency, logger_name=logger_name) for wid in range(workers))

    ## Reset all warps and set appropriate mode
    warp_list = [base, pLLC, field, shared, full_LRU]
    for w in warp_list:
        for t in w:
            t.reset()
            # Set each warp in its respective mode
            # Mode can take the following vals: 0 = independant (default), 1 = psuedo LLC en, 2 = only field view shared, 3 = All shared, 4 = All shared and set limiter enabled
            if (t.name == "base"):
                t.set_mode(0)
            elif (t.name == "pLLC"):
                t.set_mode(1)
            elif (t.name == "field"):
                t.set_mode(2)
            elif (t.name == "shared"):
                t.set_mode(3)
            elif (t.name == "full_LRU"):
                t.set_mode(4)

    run_workers(shared, field, pLLC, base, full_LRU, workers, contexts, ctxt_fname, dis_shared=dis_shared, dis_field=dis_field, dis_pLLC=dis_pLLC, dis_base=dis_base, dis_full_LRU=dis_full_LRU, logger_name=logger_name)

    return 0

"""
Main Routine
"""
def _cosim(wsl_en=0):

    ## Test default parameters
    base_assoc = 4
    test_params = {
        'workers'           : 1,        # No. of cores
        'contexts'          : 4,        # No. of threads/pages per core
        'words_per_line'    : 16,       # No. of 4-byte words per line
        'assoc'             : 4,        # Associativity of field table
        'sets'              : 384,       # Sets in shared table
        'page_size'         : 128        # Size of a single CB page
    }

    ## Test knobs
    external_access_latency = 8
    streamContext = 0           # Uniform streaming instead of interleaved
    debug = 0
    disable_base_warp = 0
    disable_pLLC_warp = 0
    disable_field_warp = 0
    disable_shared_warp = 0
    disable_full_LRU_warp = 0
    increamental_run = 0
    increamental_key = 'sets'     # Valid values: None | any key from 'test_params' dict
    increamental_max = 512        # Max value of 'increamental_key'
    increamental_step = 0       # increamental_key is stepped by increamental_step till increamental_max -> when 0, multiply by 2

    ## Run control parameters
    strm_pattern = 'stream' if (streamContext == 1) else 'interleaved'
    run_type = 'debug' if (debug == 1) else 'run'
    version = 'v3.0.0'
    ctxt_type = 'streamA' if (streamContext == 1) else 'interleavedA'       # Valid types -> streamA, interleavedA
    ctxt_path = '../sims/context/' if (wsl_en) else '..\\sims\\context\\'
    log_path = 'cosim/logs/' if (wsl_en) else 'cosim\\logs\\'
    run_name = 'cb_test'

    ## Logger
    # 'ra' = Random Policy
    log_run_name = run_name + '_' + 'wo' + str(test_params['workers']) + 'ct' + str(test_params['contexts']) + 'as' + str(test_params['assoc']) + 'se' + str(test_params['sets']) + 'pg' + str(test_params['page_size']) + 'ra1' + 'ir' + str(increamental_run) + '_' + version + '_' + strm_pattern + '_' + run_type
    logger = logging.getLogger(log_run_name)
    log_fname = log_path + log_run_name + '.log'
    log_fpath = os.path.join(os.getcwd(), log_fname)
    log_level = logging.DEBUG if (debug == 1) else logging.INFO
    logging.basicConfig(filename=log_fpath, style="{", filemode='w')
    logger.setLevel(log_level)

    ## Batch Start
    run_count = 0
    #run_access_red_avg = 0
    batch_start_time = time.time_ns()
    while True:
        ## Bandit params
        workers = test_params['workers']
        contexts = test_params['contexts']
        words_per_line = test_params['words_per_line']
        assoc = test_params['assoc']
        sets = test_params['sets']
        page_size = test_params['page_size']
        ## Baseline params
        #baseline_assoc = base_assoc
        full_sets = int(contexts*page_size/base_assoc)

        ## Matrix parameters
        """
        Batch run
        """
        mat_name = [
                'bcsstk10',
                'bcsstk13',
                'bcsstk17',
                'c8_mat11',
                'cq9',
                'fv1',
                'kl02',
                'lhr34c',
                'pdb1HYS',
                'psmigr_1',
                'wiki-Vote',
                'ca-HepTh',
                'p2p-Gnutella04',
                'as-735',
                'amazon0312',
                'Chem97ZtZ',
                'airfoil1',
                'diag',
                'crack',
                'shock-9',
                'big_dual'
            ]
        """
        Individual run
        """
        #mat_name = ['bcsstk13']
        
        logger.info(f'{_cosim.__name__} ||| ------ COSIM CONFIG [{run_count}] ------')
        logger.info(f'{_cosim.__name__} ||| Test Run Knobs: StreamContext = {streamContext}, debug_run = {debug}')
        logger.info(f'{_cosim.__name__} ||| Test Run Knobs: Increamental run = {increamental_run}, key = {increamental_key}, step = {increamental_step}, max_val = {increamental_max}')
        logger.info(f'{_cosim.__name__} ||| CacheBandit info: assoc = {assoc}, field_sets = {sets}, Full_sets = {full_sets}, workers = {workers}, pages = {contexts}, page_size = {page_size}')
        logger.info(f'{_cosim.__name__} ||| CacheBandit info: Physical shared memory (among workers) = {workers*contexts*page_size}, Virtual shared memory (among workers) = {assoc*sets}\n\n')
        
        ## Start Batch
        #cb_access_red = []
        #cb_mat_name = []
        #cb_access_red_avg = 0
        total_run_time = 0
        for mid, mname in enumerate(mat_name):
            strmA_fname = ctxt_path + ctxt_type + '_' + run_name + '_' + mname + '.csv'
            strmA_fpath = os.path.join(os.getcwd(), strmA_fname)

            matA_header = []
            with open (strmA_fpath, newline='') as rdf:
                reader = csv.reader(rdf, quoting=csv.QUOTE_NONE)
                matA_header = next(reader)

            print (f'\n{_cosim.__name__} ||| --------- RUN_START ---------')
            run_start_time = time.time_ns()
            ## Print config
            print (f'{_cosim.__name__} ||| Streaming: Matrix A dim K = {matA_header[0]}, M = {matA_header[1]}, NNZ = {matA_header[2]}, Density = {matA_header[3]}, Unique frames visited = {matA_header[4]}')
            logger.info(f'{_cosim.__name__} ||| ------ STREAM CONFIG ------')
            logger.info(f'{_cosim.__name__} ||| Streaming (type="{ctxt_type}"): Matrix {mname} ==> dim K = {matA_header[0]}, M = {matA_header[1]}, NNZ = {matA_header[2]}, Density = {matA_header[3]}, Unique frames visited = {matA_header[4]}')
            #logger.info(f'{_cosim.__name__} ||| Memory Share Participants (workers/pages participating in memory sharing) = {mem_share_en[mid]}\n')
            #logger.info(f'{_cosim.__name__} ||| Memory Share Participants (workers/pages participating in memory sharing) = {mem_share_en[0]}\n')
            ## Run CB and baseline tests
            #reduction = cacheBandit_test(workers, contexts, strmA_fname, words_per_line, assoc, sets, page_size, baseline_assoc, cb_mem_shared=mem_share_en[mid], disable_base=disable_baseline, disable_bandit=disable_bandit, llc_en=llc_en, logger_name=log_run_name)
            reduction = cacheBandit_test(workers, contexts, strmA_fname, words_per_line, assoc, sets, page_size, external_access_latency=external_access_latency, dis_shared=disable_shared_warp, dis_field=disable_field_warp, dis_pLLC=disable_pLLC_warp, dis_base=disable_base_warp, dis_full_LRU=disable_full_LRU_warp, logger_name=log_run_name)
            #cb_access_red.append(reduction)
            #cb_mat_name.append(mname)
            #cb_access_red_avg = (cb_access_red_avg+reduction)/len(cb_access_red) if (reduction is not None) else 0
            #logger.info(f'{_cosim.__name__} ||| Summary: mat_name = {cb_mat_name}, reduction = {cb_access_red}')
            
            ## Get run stats
            run_end_time = time.time_ns()
            run_time = ((run_end_time - run_start_time) * 10**(-9)) / 60
            total_run_time += run_time
            print (f'{_cosim.__name__} ||| --------- RUN_END | Run_time = {run_time} min ---------\n')
        
        ## Batch complete
        #print(f'{_cosim.__name__} ||| COSIM COMPLETE [{run_count}]: Total run time = {total_run_time} min | Access Reduction avg. = {cb_access_red_avg}\n\n')
        #logger.info(f'{_cosim.__name__} ||| COSIM COMPLETE [{run_count}]: Total run time = {total_run_time} min | Access Reduction avg. = {cb_access_red_avg}\n\n')
        print(f'{_cosim.__name__} ||| COSIM COMPLETE [{run_count}]: Total run time = {total_run_time} min\n\n')
        logger.info(f'{_cosim.__name__} ||| COSIM COMPLETE [{run_count}]: Total run time = {total_run_time} min\n\n')

        ## Batch Stats
        #run_access_red_avg = (run_access_red_avg+cb_access_red_avg)/(run_count+1)
        
        ## Check increamental run end
        cur_key = increamental_key
        cur_key_val = test_params[cur_key] if (cur_key is not None) else None
        key_val_max = increamental_max
        key_val_step = increamental_step
        if (increamental_run == 1 and cur_key_val is not None and cur_key_val < key_val_max):
            new_key_val = 2*cur_key_val if (key_val_step == 0) else cur_key_val+key_val_step
            test_params[cur_key] = new_key_val
            run_count += 1
        else:
            batch_end_time = time.time_ns()
            batch_run_time = ((batch_end_time-batch_start_time) * 10**(-9)) / 60
            #print(f'{_cosim.__name__} ||| BATCH COMPLETE: Total run time = {batch_run_time} min | Access Reduction avg. = {run_access_red_avg}\n\n')
            #logger.info(f'{_cosim.__name__} ||| BATCH COMPLETE: Total run time = {batch_run_time} min | Access Reduction avg. = {run_access_red_avg}\n\n')
            print(f'{_cosim.__name__} ||| BATCH COMPLETE: Total run time = {batch_run_time} min\n\n')
            logger.info(f'{_cosim.__name__} ||| BATCH COMPLETE: Total run time = {batch_run_time} min\n\n')
            break

if __name__ == "__main__":
    wsl_en = int(input('Running in wsl? (0/1): '))
    mode = int(input('Enter mode (0 = cosim, 1 = test): '))
    if (mode):
        test_type = int(input('Enter test type (0 = step, 1 = auto): '))
        _test(test_type)
    else:
        _cosim(wsl_en=wsl_en)
