import math
import random
import logging

class page:

    def __init__(self, page_id, page_size, pipe_len, logger_name=""):
        self.logger = logging.getLogger(logger_name)
        self.pid = page_id
        self.psize = page_size
        self.pipe_len = pipe_len
        # Page
        self.page = [x for x in range(self.psize)]                      # [Physical loc]    | page is a glorified LFSR whose taps are dynamic and time dependent, set depending on hit/miss @ time t.
        self.rev_ptr = [[x, 1] for x in range(self.psize)]                   # [Field_table_loc, ptr_null] | ptr_null = 0 -> line is on field_table; 1 -> line is on private_table/unmapped
        ## Control regs
        self.r_cb_page_rd_buffer = self.page[0-self.pipe_len-1:]                                   # Holds the available mem_locs for consumption
    
    def reset(self):
        ## Page
        self.page = [x for x in range(self.psize)]
        self.rev_ptr = [[x, 1] for x in range(self.psize)]
        ## Control regs
        self.r_cb_page_rd_buffer = self.page[0-self.pipe_len-1:]        # Holds the available mem_locs for consumption

    def print_internals(self):
        self.logger.info(f'{page.print_internals.__name__}[{self.pid}] || page: {self.page}')
        self.logger.info(f'{page.print_internals.__name__}[{self.pid}] || rev_ptr: {self.rev_ptr}')
    
    def update_page(self, page_consume):
        vloc = random.randint(0, self.psize-self.pipe_len-2)
        #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || r_cb_page_pipe = {r_cb_page_pipe}, exec_ptag = {exec_ptag}, member_id = {exec_member_id}, rand_en = {rand_en}')
        #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || vloc = {vloc}, dead = {dead}, page_consume = {page_consume}, insert_ptag = {tptag}, vtp_offset = {vtp_offset}')
        #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || page_util = {self.page_util}, page_almost_full = {self.page_almost_full}')
        self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || vloc[{vloc}], page_consume[{page_consume}]')
        # Generate rotated_page
        if (page_consume):
            rot_page = self.page[1:] + [self.page[0]]
            mask_page = self.page[:vloc] + rot_page[vloc:]
            new_page = mask_page[:-1] + [rot_page[vloc-1]]
            old_page = self.page
            self.page = new_page
            #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || old_page = {old_page}')
            #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || rot_page = {rot_page}')
            #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || mask_page = {mask_page}')
            #self.logger.debug(f'{page.update_page.__name__}[{self.pid}] || new_page = {self.page}')
    
    def decode_page(self):
        self.r_cb_page_rd_buffer = self.page[0-self.pipe_len-1:]
    
class bandit:

    def __init__(self, group_id: int, group_config: config, no_score: int = 0, logger_name: str = "") -> None:
        ## Cosim
        self.cfg = group_config
        self.params = self.cfg.get_params()
        self.regs = self.cfg.get_regs()
        self.scoreboard = self.cfg.get_nets()
        ## bandit
        self.bid = group_id
        self.words_per_line = self.params['WORDS_PER_LINE']
        self.assoc = self.params['ASSOC']
        self.sets = self.params['SETS']
        self.bsize = self.params['PAGES']
        self.page_size = self.params['PAGE_SIZE']
        self.data_w = self.params['DATA_W']
        self.limited_sets = int(self.bsize*self.page_size/self.assoc) if (int(self.bsize*self.page_size/self.assoc) <= self.sets) else self.sets
        self.priv_sets = int(self.page_size/self.assoc)
        self.pipe_len = self.params['PIPE_DEP']                                             # Number of stages in CB
        self.no_score = no_score
        self.logger = logging.getLogger(logger_name)
        self.bmember = list(page(pid, self.page_size, self.pipe_len, logger_name=logger_name) for pid in range(self.bsize)) if (self.bsize is not None and self.bsize > 0) else None
        self.tdm_id = 0
        self.share_id = 0
        self.mode = [0 for _ in range(self.bsize)]
        self.share_en = [0 for _ in range(self.bsize)]
        self.limit_en = [0 for _ in range(self.bsize)]
        self.cur_participant = 0
        self.participants = 0
        self.init_consume = [0 for _ in range(self.bsize)]               # Use this to assign values in consume buffer | increament on consume till page_size | Reset on mode 1 -> 0 change
        self.page_full = [0 for _ in range(self.bsize)]
        ## params
        self.addr_line_w = int(math.ceil(math.log(self.words_per_line, 2)))
        self.addr_assoc_w = int(math.ceil(math.log(self.assoc, 2)))
        self.addr_set_w = int(math.ceil(math.log(self.sets, 2)))
        self.limit_set_w = int(math.ceil(math.log(self.limited_sets, 2)))
        self.priv_set_w = int(math.ceil(math.log(self.priv_sets, 2)))
        self.ptag_w = int(math.ceil(math.log(self.page_size, 2)))
        self.atag_w = self.data_w - self.ptag_w
        #self.vtag_w = self.addr_set_w + self.addr_assoc_w
        self.flru_nodes = self.assoc-1
        ## Virtual Field
        self.total_sets = self.sets + (self.bsize*self.priv_sets)
        self.field_w = int(math.ceil(math.log(self.total_sets, 2)))
        self.field_size = 2**self.field_w
        self.field = [[[0, 0, 0, 0] for _ in range(self.assoc)] for _ in range(self.field_size)]    # [key, physical_loc]
        self.fflags = [[[0, 0] for _ in range(self.assoc)] for _ in range(self.field_size)]         # [vld, dirty]
        self.flru = [[0, 0] for _ in range(self.field_size)]                                        # [bit_map, LRU]
        ## Flow control
        self.r_cb_consume_dec_flag = [0 for _ in range(self.bsize)]
        self.r_cb_consume_sel = [0 for _ in range(self.bsize)]
        # Debug
        #self.print_internals()
        if (self.bmember is None):
            self.logger.error(f'{bandit.__init__.__name__}[{self.bid}] ||| Bandit Group creation faild: "group_size" parameter must be non-zero and positive')
        else:
            self.logger.info(f'{bandit.__init__.__name__}[{self.bid}] ||| Bandit Group created with bid = {self.bid}, bsize = {self.bsize}, total_sets = {self.total_sets}, field_size = {self.field_size}, page_ids = {[p.pid for p in self.bmember]}')

    def reset(self):
        ## bandit
        self.tdm_id = 0
        self.share_id = 0
        self.mode = [0 for _ in range(self.bsize)]
        self.share_en = [0 for _ in range(self.bsize)]
        self.limit_en = [0 for _ in range(self.bsize)]
        self.cur_participant = 0
        self.participants = 0
        self.init_consume = [0 for _ in range(self.bsize)]
        self.page_full = [0 for _ in range(self.bsize)]
        ## page
        _ = [p.reset() for p in self.bmember]
        ## virtual Field
        self.field = [[[0, 0, 0, 0] for _ in range(self.assoc)] for _ in range(self.field_size)]
        self.fflags = [[[0, 0] for _ in range(self.assoc)] for _ in range(self.field_size)]
        self.flru = [[0, 0] for _ in range(self.field_size)]
        ## Stage regs
        del self.regs
        self.regs = self.cfg.get_regs()
        ## scoreboard
        del self.scoreboard
        self.scoreboard = self.cfg.get_nets()
        ## Flow control
        self.r_cb_consume_dec_flag = [0 for _ in range(self.bsize)]
        self.r_cb_consume_sel = [0 for _ in range(self.bsize)]
    
    def join_shared(self, tdm_id, share_en=0, limit_en=0):
        self.logger.debug(f'{bandit.join_shared.__name__}[{self.bid}] ||| Page[{tdm_id}] joining Shared Heap, with shared_page_en = {share_en}')
        self.mode[tdm_id] = 1
        self.share_en[tdm_id] = share_en
        self.limit_en[tdm_id] = limit_en
        self.participants += share_en
        if (self.participants > self.bsize):
            self.logger.error(f'{bandit.join_shared.__name__}[{self.bid}] ||| Page[{tdm_id}] Exception! Participants in shared memory pool is greater than the number of pages in pool')
            raise Exception(f"{bandit.join_shared.__name__}[{self.bid}] ||| Page[{tdm_id}] | Participants in shared memory pool is greater than the number of pages in pool")
    
    def leave_shared(self, tdm_id):
        self.logger.debug(f'{bandit.join_shared.__name__}[{self.bid}] ||| Page[{tdm_id}] leaving Shared Heap')
        self.mode[tdm_id] = 0
        self.share_en[tdm_id] = 0
        self.participants -= 1
        self.init_consume[tdm_id] = 0
        self.page_full[tdm_id] = 0
        if (self.participants < 0):
            self.logger.error(f'{bandit.join_shared.__name__}[{self.bid}] ||| Page[{tdm_id}] Exception! Participants in shared memory pool cannot be negative')
            raise Exception(f"{bandit.join_shared.__name__}[{self.bid}] ||| Page[{tdm_id}] | Participants in shared memory pool cannot be negative")

    def print_internals(self):
        self.logger.info(f'{bandit.print_internals.__name__}[{self.bid}] || field: {self.field}')
        self.logger.info(f'{bandit.print_internals.__name__}[{self.bid}] || flru: {self.flru}')
        _ = [p.print_internals for p in self.bmember]

    def get_reg(self, reg_name):
        return self.regs[reg_name]
    
    def get_score(self, score_name):
        return self.scoreboard[score_name][0]
    
    def set_score(self, score_name, score_val):
        if (self.no_score == 0):
            self.scoreboard[score_name][0] = score_val
    
    def get_scoreboard(self):
        score = {}
        for k, v in self.scoreboard.items():
            score[k] = v[0]
        return score

    def masker(self, width):
        return 2**width-1
    
    def gen_lru_bitmap(self, bit_map, line, flip_en=0):
        node = 0            # Root node
        stages = self.addr_assoc_w
        #self.logger.debug(f'{bandit.gen_lru_bitmap.__name__}[{self.bid}] || bitmap = {bit_map:4b}, line = {line}')
        for s in range(stages):
            # get current stage decision bit
            line_bit_sel = stages-1-s
            line_bit = (line >> line_bit_sel) & 0x1
            # get current bit at node
            bit_sel = 1 << node
            bit_mask = bit_map & bit_sel
            cur_bit = bit_mask >> node
            # Update bit_map
            bit_map = bit_map ^ bit_sel if (line_bit == cur_bit and flip_en == 1) else bit_map
            # Goto next child
            node = 2*node+2 if (line_bit) else 2*node+1
        return bit_map
    
    def lru_dfs(self, bit_map):
        node = 0            # Root node
        stages = self.addr_assoc_w
        line = 0
        #self.logger.debug(f'{bandit.lru_dfs.__name__}[{self.bid}] || bitmap = {bit_map:4b}, flip_en = {flip_en}')
        for s in range(stages):
            # get current bit at node
            bit_sel = 1 << node
            bit_mask = bit_map & bit_sel
            cur_bit = bit_mask >> node
            # Set line bit for current bit at node
            line_sel = 1 << (stages-1-s)
            line_mask = line_sel if (cur_bit) else ~line_sel
            line = line | line_mask if (cur_bit) else line & line_mask
            # Goto next child
            node = 2*node+2 if (cur_bit) else 2*node+1
            #self.logger.debug(f'{bandit.lru_dfs.__name__}[{self.bid}] || STAGE[{s}] | cur_bit[{cur_bit:b}], line_sel[{line_sel:4b}], line_mask[{line_mask:4b}], line[{line:4b}]')
        return line

    def put_lru(self, set_sel, line_sel, wb_en):
        set_lru = self.flru[set_sel]
        bit_map = set_lru[0]
        #self.logger.debug(f'{bandit.put_lru.__name__}[{self.bid}] || cur_lru_bitmap = {bit_map:4b}, cur_lru = {line} | wb_en = {wb_en}')
        # Update bit_map
        new_bit_map = self.gen_lru_bitmap(bit_map, line_sel, flip_en=wb_en)
        # Update lru
        new_lru_line = self.lru_dfs(new_bit_map)
        wb_val = [new_bit_map, new_lru_line]
        #self.logger.debug(f'{bandit.put_lru.__name__}[{self.bid}] || nxt_lru_bitmap = {new_bit_map:4b}, nxt_lru = {new_line}')
        # Writeback
        self.flru[set_sel] = wb_val
        # score
        self.set_score('w_cb_lru_nxt_bits', new_bit_map)
        self.set_score('w_cb_lru_nxt', new_lru_line)
        self.set_score('w_cb_lru_cur_bits', bit_map)
        #return lru_line
    
    def decode_page(self, page_sel):
        self.bmember[page_sel].decode_page()

    def update_page(self, page_consume, page_sel):
        self.bmember[page_sel].update_page(page_consume)

    def get_lru(self, page_sel, set_sel, direct_en):
        set_lru = self.flru[set_sel]
        lru_line = page_sel if (direct_en) else set_lru[1]
        self.set_score('w_cb_lru_cur', lru_line)
        return lru_line

    def get_field_table(self, set_sel):
        field_table = self.field[set_sel]
        #self.logger.debug(f'{bandit.get_field_table.__name__} ||| Field for set[{set_sel}] = {field_table}')
        return field_table

    def get_validity_flag(self, set_sel):
        vld_flag = [f[0] for f in self.fflags[set_sel]]
        #self.logger.debug(f'{bandit.get_validity_flag.__name__} ||| pvlds for set[{set_sel}] = {vld_table}')
        return vld_flag
    
    def get_dirty_flag(self, set_sel, cb_cline, cb_rev_ptr, cb_consume, direct_en):
        rev_set_sel = cb_rev_ptr >> self.addr_assoc_w
        rev_line = self.masker(self.addr_assoc_w) & cb_rev_ptr
        dirty_set_sel = rev_set_sel if (cb_consume and direct_en == 0) else set_sel
        dirty_line = rev_line if (cb_consume and direct_en == 0) else cb_cline
        dirty_flag = self.fflags[dirty_set_sel][dirty_line][1]
        #self.logger.debug(f"{bandit.get_dirty_flag.__name__}[{self.bid}] || dirty_set = {dirty_set_sel}, dirty_line = {dirty_line}, dirty_loc = {dirty_loc}, dirty_flag = {dirty_flag}")       
        return dirty_flag

    def get_rev_ptr(self, abs_phy_loc, shared_sel, set_sel, lru_line, shared):
        # Get rev_ptr @ phy_loc
        rel_set_sel = set_sel & self.masker(self.priv_set_w)
        rel_phy_loc = abs_phy_loc & self.masker(self.ptag_w)
        priv_phy_loc = (rel_set_sel << self.addr_assoc_w) | lru_line
        phy_loc = rel_phy_loc if (shared) else priv_phy_loc
        rev_ptr = self.bmember[shared_sel].rev_ptr[phy_loc]
        field_loc = rev_ptr[0]
        field_null = rev_ptr[1]
        self.logger.debug(f'{bandit.get_rev_ptr.__name__}[{self.bid}][{shared_sel}] || Got Field_loc, field_null = ({field_loc}, {field_null}) @ phy_loc[{phy_loc}]')
        # score
        self.set_score('w_cb_rev_ptr', field_loc)
        self.set_score('w_cb_rev_ptr_null', field_null)
        return field_loc, field_null
    
    def get_eviction(self, set_sel, cb_cline, cb_rev_ptr, cb_consume, direct_en):
        delim = self.bsize*self.priv_sets
        rev_set_sel = cb_rev_ptr >> self.addr_assoc_w
        rev_line = self.masker(self.addr_assoc_w) & cb_rev_ptr
        evict_set_sel = rev_set_sel if (cb_consume and direct_en == 0) else set_sel
        evict_line = rev_line if (cb_consume and direct_en == 0) else cb_cline
        out_set = ~(evict_set_sel) & self.masker(self.field_w) if (evict_set_sel > delim) else evict_set_sel
        out_set_w = self.addr_set_w if (evict_set_sel > delim) else self.priv_set_w
        evict_atag = self.field[evict_set_sel][evict_line][0]
        evict_ptag = self.field[evict_set_sel][evict_line][1]
        heap_rd_addr = evict_ptag
        dma_wr_addr = ((evict_atag << out_set_w) | out_set) << self.addr_assoc_w
        #self.logger.debug(f"{bandit.check_field.__name__}[{self.bid}] || eset = {evict_set_sel}, eline = {evict_line}, oset = {out_set}, consume = {cb_consume} | heap_rd_addr = {heap_rd_addr}, dma_wr_addr = {dma_wr_addr}")
        return heap_rd_addr, dma_wr_addr
    
    def put_rev_ptr(self, set_sel, cb_cline, abs_phy_loc, cb_consume, shared_sel, direct_en, shared):
        # Writeback field_loc @ phy_loc when consume
        rel_set_sel = set_sel & self.masker(self.priv_set_w)
        data_msb = set_sel << self.addr_assoc_w
        data_lsb = cb_cline
        rev_ptr_wr_data = data_msb | data_lsb
        priv_msb = rel_set_sel << self.addr_assoc_w
        priv_lsb = cb_cline
        rel_priv_loc = priv_msb | priv_lsb
        rel_phy_loc = abs_phy_loc & self.masker(self.ptag_w)
        phy_loc = rel_phy_loc if (shared == 1) else rel_priv_loc
        self.logger.debug(f'{bandit.put_rev_ptr.__name__}[{self.bid}][{shared_sel}] || cb_consume = {cb_consume}, direct_en = {direct_en}')
        if (cb_consume == 1 and direct_en == 0):
            null = 1 if (shared == 0) else 0
            self.bmember[shared_sel].rev_ptr[phy_loc][0] = rev_ptr_wr_data
            self.bmember[shared_sel].rev_ptr[phy_loc][1] = null
            self.logger.debug(f'{bandit.put_rev_ptr.__name__}[{self.bid}][{shared_sel}] || Wrote Field_loc, null = {rev_ptr_wr_data}, {null} @ phy_loc[{phy_loc}]')
    
    def put_field_table(self, set_sel, cb_cline, atag, cb_ctag, phy_buf_loc, cb_miss, cb_consume, shared):
        field_table_wr_data = phy_buf_loc if (cb_consume) else cb_ctag
        field_table = self.field[set_sel]
        if (cb_miss):
            field_table[cb_cline][0] = atag
            if (shared == 1):
                field_table[cb_cline][1] = field_table_wr_data
            #self.logger.debug(f'{bandit.put_field_table.__name__}[{self.bid}] || Updated field_table @ set_sel[{set_sel}], cb_cline[{cb_cline}] = {field_table} | atag = {self.field[set_sel][cb_cline][0]}, ptag = {self.field[set_sel][cb_cline][1]} | shared = {shared}')
    
    def put_validity_table(self, set_sel, cb_cline, cb_rev_ptr, cb_rev_ptr_null, cb_consume, direct_en):
        # Clear valid flag
        clr_vld_fset_sel = set_sel if (direct_en) else cb_rev_ptr >> self.addr_assoc_w
        clr_vld_fline = cb_cline if (direct_en) else self.masker(self.addr_assoc_w) & cb_rev_ptr
        clr_vld_fen = 1 if ((cb_consume and cb_rev_ptr_null == 0) or direct_en) else 0
        if (clr_vld_fen):
            self.fflags[clr_vld_fset_sel][clr_vld_fline][0] = 0
            self.logger.debug(f'{bandit.put_validity_table.__name__}[{self.bid}] || consume = {cb_consume}, clr_vld_fen = {clr_vld_fen} | Field validity_table @ set[{clr_vld_fset_sel}], line[{clr_vld_fline}] = {self.fflags[clr_vld_fset_sel][clr_vld_fline][0]}')
        # Set valid flag
        set_vld_fset_sel = set_sel
        set_vld_fline = cb_cline
        set_vld_fen = 1 if (cb_consume) else 0
        if (set_vld_fen):
            self.fflags[set_vld_fset_sel][set_vld_fline][0] = 1
            self.logger.debug(f'{bandit.put_validity_table.__name__}[{self.bid}] || consume = {cb_consume}, set_vld_fen = {set_vld_fen} | Field validity_table @ set[{set_vld_fset_sel}], line[{set_vld_fline}] = {self.fflags[set_vld_fset_sel][set_vld_fline][0]}')
        #self.logger.debug(f'{bandit.put_validity_table.__name__}[{self.bid}] || Field validity_table @ set[{clr_vld_fset_sel}] = {self.field[clr_vld_fset_sel]}')
    
    def put_dirty_table(self, mem_wen, set_sel, cb_cline, cb_rev_ptr, cb_miss, cb_consume, cb_en, direct_en):
        # Clear Dirty flag
        clr_dty_fset_sel = cb_rev_ptr >> self.addr_assoc_w if (cb_consume and direct_en == 0) else set_sel
        clr_dty_fline = self.masker(self.addr_assoc_w) & cb_rev_ptr if (cb_consume and direct_en == 0) else cb_cline
        clr_dty_fen = 1 if (cb_miss or direct_en) else 0
        if (clr_dty_fen):
            self.fflags[clr_dty_fset_sel][clr_dty_fline][1] = 0
            #self.logger.debug(f'{bandit.put_dirty_table.__name__}[{self.bid}] || miss = {cb_miss}, consume = {cb_consume}, clr_dty_fen = {clr_dty_fen} | Field dirty_table @ set[{clr_dty_fset_sel}], line[{clr_dty_fline}] = {self.fflags[clr_dty_fset_sel][clr_dty_fline][1]}')
        # Set Dirty flag
        set_dty_fset_sel = set_sel
        set_dty_fline = cb_cline
        set_dty_fen = 1 if (mem_wen and cb_en and cb_miss == 0 and direct_en == 0) else 0
        if (set_dty_fen):
            self.fflags[set_dty_fset_sel][set_dty_fline][1] = 1
            #self.logger.debug(f'{bandit.put_dirty_table.__name__}[{self.bid}] || mem_wen = {mem_wen}, cb_en = {cb_en}, miss = {cb_miss}, set_dty_fen = {set_dty_fen} | Field dirty_table @ set[{set_dty_fset_sel}], line[{set_dty_fline}] = {self.fflags[set_dty_fset_sel][set_dty_fline][1]}')
    
    def check_field(self, set_sel, atag, field_table, set_vld_flag, flru_line, cb_en, ignore_miss, direct_en, shared):
        gen_rel_ptag = lambda y, z: (y << self.addr_assoc_w) | z
        self.logger.debug(f"{bandit.check_field.__name__}[{self.bid}] || flru_line = {flru_line}, Field_table = {field_table}, vld_table = {set_vld_flag} | Shared = {shared}")
        default_line = flru_line
        default_ptag = field_table[flru_line][1] if (shared == 1) else gen_rel_ptag(set_sel, flru_line)
        flru_pvld = set_vld_flag[flru_line]
        consume = 0
        ## Compare tags
        for lid, line in enumerate(field_table):
            hit_vld = 1 if (line[0] == atag and set_vld_flag[lid] and cb_en and direct_en == 0) else 0
            miss_vld = 0 if ((line[0] == atag and set_vld_flag[lid]) or ignore_miss or direct_en or cb_en == 0) else 1
            hit = hit_vld
            miss = miss_vld
            comp_line = lid if (hit_vld) else default_line
            comp_ptag = line[1] if (hit_vld) else default_ptag
            consume = 1 if (miss_vld and flru_pvld == 0) else 0
            if (hit_vld):
                #self.logger.debug(f"{bandit.check_field.__name__}[{self.bid}] || HIT for key[{atag}] at line[{comp_line}], ptag = {comp_ptag} ; in set[{set_sel}]")
                break
        ## Miss branch
        if (miss):
            self.logger.debug(f"{bandit.check_field.__name__}[{self.bid}] || MISS for key[{atag}] in set[{set_sel}] | lru_line = {comp_line}, lru_ptag = {comp_ptag}, consume = {consume} | cb_en = {cb_en}, ignore_miss = {ignore_miss}, shared = {shared}")
        
        field_stall = lambda x: 0 if (x == 1) else 1
        self.set_score('w_field_stall', field_stall(cb_en))
        self.set_score('w_cb_hit', hit)
        self.set_score('w_cb_miss', miss)
        self.set_score('w_cb_cline', comp_line)
        self.set_score('w_cb_ctag', comp_ptag)
        self.set_score('w_cb_consume', consume)
        
        return comp_line, comp_ptag, hit, miss, consume

    def controller(self, consume, tdm_sel, shared_sel, direct_en, shared, share_en):
        # get regs
        r_cb_dec_flag = self.r_cb_consume_dec_flag[shared_sel]
        consume_sel = self.r_cb_consume_sel[shared_sel]
        # Move pipe
        r_cb_dec_flag = (r_cb_dec_flag << 1) & self.masker(self.pipe_len+1) 
        # Extract dec and update flags
        dec_flag = (r_cb_dec_flag >> self.pipe_len) & 0x1
        update_flag = (r_cb_dec_flag >> (self.pipe_len-1)) & 0x1
        update_share = 1 if (consume == 1 and shared == 1 and share_en == 1 and direct_en == 0) else 0
        self.logger.debug(f'{bandit.controller.__name__}[{self.bid}][{shared_sel}] || flag_pipe = {r_cb_dec_flag:0b}, dec_flag = {dec_flag}, update_flag = {update_flag}, cb_consume_sel = {consume_sel}')
        # Compute new control vals
        t_cb_dec_flag = r_cb_dec_flag | consume
        t_consume_sel = consume_sel + consume - dec_flag
        # score
        self.set_score('w_cb_consume_sel', consume_sel)
        # update regs
        self.r_cb_consume_dec_flag[shared_sel] = t_cb_dec_flag
        self.r_cb_consume_sel[shared_sel] = t_consume_sel
        self.page_full[shared_sel] = 1 if (self.init_consume[shared_sel] == self.page_size-1 and consume == 1 and direct_en == 0) else self.page_full[shared_sel]
        self.init_consume[shared_sel] = self.init_consume[shared_sel]+1 if (shared == 1 and self.page_full[shared_sel] == 0 and consume == 1 and direct_en == 0) else self.init_consume[shared_sel]
        self.tdm_id = tdm_sel+1 if (tdm_sel != self.bsize-1) else 0
        if (update_share):
            self.logger.debug(f'{bandit.controller.__name__}[{self.bid}] ||| Update share = {update_share}, cur_patricipant = {self.cur_participant} | share_en = {self.share_en}')
            # Find next participant, i.e., xth 1 in mode list where x is current participant+1
            nxt_participant = self.cur_participant+1 if (self.cur_participant != self.participants-1) else 0
            found_parts = 0
            found_at = shared_sel
            for xid, x in enumerate(self.share_en):
                if (x == 1):
                    if (found_parts < nxt_participant):
                        found_parts += 1
                    else:
                        found_at = xid
                        break
            self.cur_participant = nxt_participant
            self.share_id = found_at
        self.logger.debug(f'{bandit.controller.__name__}[{self.bid}] ||| next_patricipant = {self.cur_participant}, next TDM_sel = {self.tdm_id}, next Shared_page_sel = {self.share_id}')
        #self.logger.debug(f'{bandit.controller.__name__}[{self.bid}][{shared_sel}] || init_consume = {self.init_consume[shared_sel]}, page_full = {self.page_full[shared_sel]} | next_tdm_channel = {self.tdm_id}')

        return update_flag, update_share

    def interface_decoder(self, channel_sel, addr, reset, bandit_en, ignore_miss, mode_switch, direct):
        # Setup
        cb_en = 0 if (reset or mode_switch) else bandit_en
        tdm_sel = self.tdm_id if (ignore_miss == 0 and direct == 0) else channel_sel
        shared = self.mode[tdm_sel]
        share_en = self.share_en[tdm_sel]
        limit_en = self.limit_en[tdm_sel]
        page_sel = self.share_id if (share_en and direct == 0) else tdm_sel
        direct_en = 1 if (direct and shared == 0) else 0

        # Extract
        taddr = addr >> self.addr_line_w                    # drop LSBs since all these words belong to the same line
        priv_set_w = self.priv_set_w
        priv_prefix = tdm_sel << priv_set_w
        priv_set_sel = (taddr & self.masker(priv_set_w)) | priv_prefix
        shared_set_w = self.limit_set_w if (limit_en) else self.addr_set_w
        shared_set_sel = ~(taddr & self.masker(shared_set_w)) & self.masker(self.field_w)     # Invert all shared_set_sel to address the field table from the other end | This avoids an adder to acheive speed
        faddr = taddr >> shared_set_w if (shared) else taddr >> priv_set_w 
        atag = faddr & self.masker(self.atag_w)
        set_sel = shared_set_sel if (shared) else priv_set_sel
        debug_set_sel = self.field_size-set_sel-1               # Print this in shared mode to obtain the normalized set_sel value
        #self.logger.debug(f'{bandit.interface_decoder.__name__}[{self.bid}] || TDM_CHANNEL = {tdm_sel} | incoming addr = {addr:#0b} | shared = {shared}, line_w = {self.addr_line_w}, set_w = {addr_set_w}')
        self.logger.debug(f'{bandit.interface_decoder.__name__}[{self.bid}] || set_sel = {set_sel}, atag = {atag} | debug_set_sel = {debug_set_sel}')
        
        # Get solution from page
        r_cb_consume_sel = self.r_cb_consume_sel[page_sel]
        r_cb_page_rd_buffer = self.bmember[page_sel].r_cb_page_rd_buffer
        self.logger.debug(f'{bandit.interface_decoder.__name__}[{self.bid}][{page_sel}] || cb_buffer_sel = {r_cb_consume_sel}, page_rd_buffer = {r_cb_page_rd_buffer} | page_sel = {page_sel}, page_full = {self.page_full[page_sel]}')
        lsb = r_cb_page_rd_buffer[r_cb_consume_sel] if (self.page_full[page_sel] == 1) else self.init_consume[page_sel]
        msb = page_sel << self.ptag_w
        phy_loc = msb | lsb                 # Generate absolute physical location for memory/heap access
        self.logger.debug(f'{bandit.interface_decoder.__name__}[{self.bid}][{page_sel}] || Current solution: rel_phy_loc = {lsb}, abs_phy_loc[{phy_loc}] | direct_en = {direct_en}, page_sel = {page_sel}')
        
        # score
        self.set_score('w_field_set', set_sel)
        self.set_score('w_field_atag', atag)

        return cb_en, tdm_sel, shared, share_en, page_sel, direct_en, set_sel, atag, phy_loc
    
    def writeback_pipe_score(self, set_sel, atag, buf_phy_loc, cb_cline, cb_ctag, cb_rev_ptr, cb_rev_ptr_null, cb_miss, cb_consume):
        if (self.no_score == 0):
            # Field table scores
            self.set_score('wo_field_table_wr_en', self.regs['r_wb_cb_miss'])
            msb = self.regs['r_wb_atag'] << self.ptag_w
            lsb = self.regs['r_wb_buf_phy_loc'] if (cb_consume) else self.regs['r_wb_cb_ctag']
            field_table_wr_data = msb | lsb
            self.set_score('wo_field_table_wr_data', field_table_wr_data)
            msb = self.regs['r_wb_set_sel'] << self.addr_assoc_w
            lsb = self.regs['r_wb_cb_cline']
            field_table_wr_addr = msb | lsb
            self.set_score('wo_field_table_wr_addr', field_table_wr_addr)
            # Rev_ptr table scores
            msb = self.regs['r_wb_set_sel'] << self.addr_assoc_w
            lsb = self.regs['r_wb_cb_cline']
            rev_ptr_wr_data = msb | lsb
            self.set_score('wo_rev_ptr_wr_addr', self.regs['r_wb_buf_phy_loc'])
            self.set_score('wo_rev_ptr_wr_data', rev_ptr_wr_data)
            self.set_score('wo_rev_ptr_wr_en', self.regs['r_wb_cb_consume'])
            # Validity table scores
            clear_en = self.regs['r_wb_cb_rev_ptr_null'] ^ 0x1
            set_en = self.regs['r_wb_cb_consume']
            msb = self.regs['r_wb_set_sel'] << self.addr_assoc_w
            lsb = self.regs['r_wb_cb_cline']
            field_ptr = msb | lsb
            self.set_score('wo_validity_set', set_en)
            self.set_score('wo_validity_set_addr', field_ptr)
            self.set_score('wo_validity_clear', clear_en)
            self.set_score('wo_validity_clear_addr', self.regs['r_wb_cb_rev_ptr'])
            # pucb_intf to CB scores
            self.set_score('wo_cb_consume', self.regs['r_wb_cb_consume'])
            # Register values
            self.regs['r_wb_set_sel'] = set_sel
            self.regs['r_wb_atag'] = atag
            self.regs['r_wb_buf_phy_loc'] = buf_phy_loc
            self.regs['r_wb_cb_cline'] = cb_cline
            self.regs['r_wb_cb_ctag'] = cb_ctag
            self.regs['r_wb_cb_miss'] = cb_miss
            self.regs['r_wb_cb_consume'] = cb_consume
            self.regs['r_wb_cb_rev_ptr'] = cb_rev_ptr
            self.regs['r_wb_cb_rev_ptr_null'] = cb_rev_ptr_null
            self.regs['r_wb_cb_consume'] = cb_consume
    
    def run(self, mode_sel=0, channel_sel=0, addr=0, wen=0, bandit_en=0, reset=0, ignore_miss=0, ignore_dma=0, mode_switch_en=0, page_share_en=0, limit_en=0, direct=0):
        
        ret_val = None

        ## NOTE:
        # A miss (IN HARDWARE) requires a minimum bubble of 1 cycle before requesting again.
        # Due to the 1 cycle writeback delay of all tables.
        #
        # Enable direct mode by setting shared = 0 and direct = 1.
        # Shared must be 0 for direct mapping of page to lines in interface_decoder.
        # Use addr to select sets (lsbs ignored) and channel_sel to select lines in direct mode.
        # A direct write will flush the line automatically
        
        ### DATAFLOW_BEGIN ###
        
        ## First Stage
        # Split address into atag (Key) and set sel and current available solution from page buffer
        cb_en, tdm_sel, shared, share_en, page_sel, direct_en, set_sel, atag, buf_phy_loc = self.interface_decoder(channel_sel, addr, reset, bandit_en, ignore_miss, mode_switch_en, direct)
        self.logger.debug(f'{bandit.run.__name__}[{self.bid}] || TDM_Channel = {page_sel}, page_sel = {page_sel}, Channels = {self.bsize} | Recieved addr[{addr}], wen[{wen}], cb_en[{cb_en}] -> set_sel = {set_sel}, atag = {atag}')
        
        ## Begin Cosim run
        self.set_score('w_cb_tdm', tdm_sel)
        self.set_score('w_field_wen', wen)

        ## Second Stage
        # Get pLRU
        flru_line = self.get_lru(page_sel, set_sel, direct_en)
        # Get tag table
        set_tag_table = self.get_field_table(set_sel)
        # Get validity table
        set_vld_flag = self.get_validity_flag(set_sel)
            
        ## Third Stage
        # Check field_table
        cb_cline, cb_ctag, cb_hit, cb_miss, cb_consume = self.check_field(set_sel, atag, set_tag_table, set_vld_flag, flru_line, cb_en, ignore_miss, direct_en, shared)
        # Get rev_ptr
        cb_rev_ptr, cb_rev_ptr_null = self.get_rev_ptr(buf_phy_loc, page_sel, set_sel, flru_line, shared)

        ## Fourth Stage
        # Get dirty flag
        cb_dirty_flag = self.get_dirty_flag(set_sel, cb_cline, cb_rev_ptr, cb_consume, direct_en)
        # Get eviction
        cb_emem_rd_addr, cb_edma_wr_addr = self.get_eviction(set_sel, cb_cline, cb_rev_ptr, cb_consume, direct_en)
        # Run controller
        cb_page_update = self.controller(cb_consume, tdm_sel, page_sel, direct_en, shared, share_en)
        # Write LRU
        self.put_lru(set_sel, cb_cline, cb_en)

        wb_addr, wb_wen, wb_cb_en = addr, wen, cb_en
        wb_set_sel, wb_atag, wb_buf_phy_loc, wb_cb_cline = set_sel, atag, buf_phy_loc, cb_cline
        wb_cb_ctag, wb_cb_rev_ptr, wb_cb_rev_ptr_null = cb_ctag, cb_rev_ptr, cb_rev_ptr_null
        wb_cb_miss, wb_cb_consume, wb_page_update = cb_miss, cb_consume, cb_page_update
        wb_cb_emem_rd_addr, wb_cb_edma_wr_addr, wb_cb_dirty_flag = cb_emem_rd_addr, cb_edma_wr_addr, cb_dirty_flag
        wb_page_sel, wb_direct_en, wb_shared = page_sel, direct_en, shared

        ## Assert writebacks based on control signals from check_field method
        #self.logger.debug(f'{bandit.run.__name__}[{self.bid}] || WRITEBACK_PHASE | Writing tables and regs')

        ## Fifth Stage | Registered writebacks
        # Cosim register Scoring
        self.writeback_pipe_score(wb_set_sel, wb_atag, wb_buf_phy_loc, wb_cb_cline, wb_cb_ctag, wb_cb_rev_ptr, wb_cb_rev_ptr_null, wb_cb_miss, wb_cb_consume)
        # Decode page (through handler) | always appears before update_page
        self.decode_page(wb_page_sel)
        # Update page (through handler)
        self.update_page(wb_page_update, wb_page_sel)
        # Write field table
        self.put_field_table(wb_set_sel, wb_cb_cline, wb_atag, wb_cb_ctag, wb_buf_phy_loc, wb_cb_miss, wb_cb_consume, wb_shared)
        # Write rev_ptr table
        self.put_rev_ptr(wb_set_sel, wb_cb_cline, wb_buf_phy_loc, wb_cb_consume, wb_page_sel, direct_en, wb_shared)
        # Write validity table
        self.put_validity_table(wb_set_sel, wb_cb_cline, wb_cb_rev_ptr, wb_cb_rev_ptr_null, wb_cb_consume, wb_direct_en)
        # Write dirty table
        self.put_dirty_table(wb_wen, wb_set_sel, wb_cb_cline, wb_cb_rev_ptr, wb_cb_miss, wb_cb_consume, wb_cb_en, wb_direct_en)
            
        ### DATAFLOW_END ###

        # Debug print
        #self.print_internals()

        # Send unregistered Memory outputs
        direct_addr = (set_sel << self.addr_assoc_w) | (channel_sel & self.masker(self.addr_assoc_w))
        mem_addr = direct_addr if (direct_en) else cb_ctag
        mem_wen = wen if (cb_hit or direct_en) else 0
        # Send Registered DMA outputs
        dma_rd_addr = wb_addr
        dma_wr_addr = wb_cb_edma_wr_addr
        heap_rd_addr = wb_cb_emem_rd_addr
        heap_wr_addr = wb_buf_phy_loc if (wb_cb_consume) else wb_cb_ctag
        heap_wr_en = 1 if (wb_cb_miss and ignore_dma != 1) else 0                   # dma_rd_en
        dma_wr_en = wb_cb_dirty_flag if (wb_cb_miss or wb_direct_en) else 0

        ## Cosim Output scoring
        self.set_score('o_mem_addr', mem_addr)
        self.set_score('o_mem_wen', mem_wen)
        self.set_score('o_dma_rd_addr', self.regs['r_dma_rd_addr'])
        self.set_score('o_heap_wr_addr', self.regs['r_heap_wr_addr'])
        self.set_score('o_dma_en', self.regs['r_dma_en'])
        # Set regs | all regs clked @ pu_clk
        self.regs['r_dma_rd_addr'] = dma_rd_addr
        self.regs['r_heap_wr_addr'] = heap_wr_addr
        self.regs['r_dma_en'] = heap_wr_en

        # cb_miss, mem_addr and mem_wen are unregistered outputs
        ret_val = (wb_cb_consume, cb_miss, cb_hit, mem_addr, mem_wen, heap_wr_en, dma_rd_addr, heap_wr_addr, dma_wr_en, dma_wr_addr, heap_rd_addr)
                
        if (reset == 1):
            self.logger.debug(f'{bandit.run.__name__}[{self.bid}] ||| Reseting Bandit!')
            self.reset()
        elif (mode_switch_en == 1):
            if (mode_sel != shared):
                _ = self.leave_shared(tdm_sel) if (mode_sel == 0) else self.join_shared(tdm_sel, page_share_en, limit_en)

        return ret_val
