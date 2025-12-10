// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval_triggers__ico(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__ico\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered.setBit(0U, (IData)(vlSelfRef.__VicoFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__ico(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ico_sequent__TOP__0\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<4>/*127:0*/ __Vtemp_1;
    VlWide<4>/*127:0*/ __Vtemp_3;
    VlWide<4>/*127:0*/ __Vtemp_4;
    VlWide<4>/*127:0*/ __Vtemp_5;
    VlWide<4>/*127:0*/ __Vtemp_6;
    // Body
    if (((IData)(vlSelfRef.pucb_intf__DOT__r_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 906, vlSelfRef.pucb_intf__DOT__r_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_mshr 
            = vlSelfRef.pucb_intf__DOT__r_field_mshr;
    }
    if ((vlSelfRef.pucb_intf__DOT__r_field_addr ^ vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_addr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 908, vlSelfRef.pucb_intf__DOT__r_field_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_addr 
            = vlSelfRef.pucb_intf__DOT__r_field_addr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__r_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 972, vlSelfRef.pucb_intf__DOT__r_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_wen 
            = vlSelfRef.pucb_intf__DOT__r_field_wen;
    }
    __Vtemp_1[0U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[0U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[0U]);
    __Vtemp_1[1U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[1U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[1U]);
    __Vtemp_1[2U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[2U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[2U]);
    __Vtemp_1[3U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[3U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[3U]);
    if (__Vtemp_1) {
        VL_COV_TOGGLE_CHG_ST_W(128, vlSymsp->__Vcoverage + 2352, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[0U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[0U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[1U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[1U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[2U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[2U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[3U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[3U];
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_evict))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3116, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_evict);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_evict 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3704, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3708, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag;
    }
    if ((vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3722, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3786, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3788, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3796, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4202, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4204, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4208, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4210, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en = 
        ((1U == (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr))
          ? ([&]() {
                ++(vlSymsp->__Vcoverage[3800]);
            }(), 1U) : ([&]() {
                ++(vlSymsp->__Vcoverage[3801]);
            }(), 0U));
    vlSelfRef.pucb_intf__DOT__o_field_mshr = vlSelfRef.pucb_intf__DOT__r_field_mshr;
    vlSelfRef.pucb_intf__DOT__o_field_addr = vlSelfRef.pucb_intf__DOT__r_field_addr;
    vlSelfRef.pucb_intf__DOT__o_field_wen = vlSelfRef.pucb_intf__DOT__r_field_wen;
    vlSelfRef.pucb_intf__DOT__i_cb_clk = vlSelfRef.i_cb_clk;
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3234, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [1U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [1U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3250, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [1U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [1U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[1U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [1U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [2U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [2U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3266, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [2U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [2U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[2U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [2U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [3U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [3U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3282, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [3U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [3U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[3U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [3U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [4U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [4U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3298, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [4U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [4U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[4U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [4U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [5U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [5U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3314, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [5U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [5U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[5U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [5U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [6U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [6U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3330, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [6U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [6U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[6U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [6U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [7U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [7U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3346, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [7U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [7U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[7U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [7U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [8U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [8U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3362, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [8U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [8U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[8U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [8U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [9U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [9U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3378, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [9U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [9U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[9U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [9U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xaU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xaU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3394, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xaU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xaU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xaU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xaU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xbU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xbU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3410, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xbU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xbU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xbU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xbU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xcU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xcU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3426, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xcU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xcU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xcU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xcU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xdU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xdU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3442, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xdU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xdU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xdU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xdU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xeU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xeU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3458, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xeU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xeU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xeU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xeU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xfU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xfU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3474, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xfU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xfU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xfU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xfU];
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag;
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2754, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [1U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [1U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2764, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [1U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [1U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[1U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [1U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [2U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [2U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2774, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [2U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [2U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[2U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [2U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [3U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [3U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2784, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [3U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [3U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[3U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [3U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [4U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [4U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2794, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [4U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [4U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[4U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [4U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [5U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [5U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2804, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [5U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [5U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[5U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [5U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [6U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [6U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2814, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [6U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [6U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[6U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [6U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [7U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [7U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2824, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [7U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [7U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[7U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [7U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [8U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [8U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2834, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [8U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [8U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[8U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [8U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [9U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [9U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2844, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [9U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [9U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[9U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [9U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xaU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xaU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2854, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xaU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xaU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xaU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xaU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xbU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xbU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2864, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xbU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xbU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xbU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xbU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xcU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xcU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2874, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xcU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xcU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xcU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xcU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xdU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xdU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2884, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xdU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xdU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xdU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xdU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xeU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xeU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2894, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xeU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xeU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xeU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xeU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xfU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xfU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2904, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xfU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xfU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xfU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xfU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x10U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x10U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2914, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x10U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x10U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x10U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x10U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x11U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x11U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2924, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x11U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x11U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x11U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x11U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x12U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x12U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2934, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x12U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x12U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x12U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x12U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x13U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x13U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2944, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x13U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x13U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x13U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x13U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x14U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x14U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2954, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x14U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x14U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x14U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x14U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x15U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x15U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2964, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x15U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x15U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x15U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x15U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x16U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x16U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2974, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x16U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x16U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x16U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x16U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x17U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x17U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2984, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x17U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x17U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x17U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x17U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x18U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x18U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2994, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x18U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x18U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x18U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x18U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x19U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x19U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3004, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x19U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x19U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x19U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x19U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1aU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1aU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3014, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1aU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1aU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1aU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1aU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1bU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1bU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3024, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1bU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1bU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1bU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1bU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1cU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1cU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3034, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1cU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1cU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1cU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1cU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1dU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1dU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3044, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1dU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1dU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1dU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1dU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1eU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1eU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3054, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1eU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1eU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1eU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1eU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1fU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1fU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3064, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1fU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1fU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1fU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1fU];
    }
    vlSelfRef.pucb_intf__DOT__i_pu_clk = vlSelfRef.i_pu_clk;
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap 
        = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg) 
                 >> 1U));
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap 
        = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg;
    vlSelfRef.pucb_intf__DOT__i_field_wen = vlSelfRef.i_field_wen;
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel 
        = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel;
    vlSelfRef.pucb_intf__DOT__i_cb_consume_buf = vlSelfRef.i_cb_consume_buf;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__i_cb_en = vlSelfRef.i_cb_en;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume;
    vlSelfRef.pucb_intf__DOT__i_rst = vlSelfRef.i_rst;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag 
        = (0xfU & vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data);
    vlSelfRef.pucb_intf__DOT__i_field_mshr = vlSelfRef.i_field_mshr;
    vlSelfRef.pucb_intf__DOT__i_field_addr = vlSelfRef.i_field_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3694, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 92, vlSelfRef.pucb_intf__DOT__o_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_mshr 
            = vlSelfRef.pucb_intf__DOT__o_field_mshr;
    }
    vlSelfRef.o_field_mshr = vlSelfRef.pucb_intf__DOT__o_field_mshr;
    if ((vlSelfRef.pucb_intf__DOT__o_field_addr ^ vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_addr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 94, vlSelfRef.pucb_intf__DOT__o_field_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_addr 
            = vlSelfRef.pucb_intf__DOT__o_field_addr;
    }
    vlSelfRef.o_field_addr = vlSelfRef.pucb_intf__DOT__o_field_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 158, vlSelfRef.pucb_intf__DOT__o_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_wen 
            = vlSelfRef.pucb_intf__DOT__o_field_wen;
    }
    vlSelfRef.o_field_wen = vlSelfRef.pucb_intf__DOT__o_field_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__i_cb_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2, vlSelfRef.pucb_intf__DOT__i_cb_clk, vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_clk);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_clk 
            = vlSelfRef.pucb_intf__DOT__i_cb_clk;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk 
        = vlSelfRef.pucb_intf__DOT__i_cb_clk;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3690, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset;
    if ((vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3608, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3674, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_ctag = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__i_pu_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_pu_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 0, vlSelfRef.pucb_intf__DOT__i_pu_clk, vlSelfRef.pucb_intf__DOT____Vtogcov__i_pu_clk);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_pu_clk 
            = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk 
        = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk 
        = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk 
        = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk 
        = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk 
        = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4200, vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4198, vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__i_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 10, vlSelfRef.pucb_intf__DOT__i_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_wen 
            = vlSelfRef.pucb_intf__DOT__i_field_wen;
    }
    vlSelfRef.pucb_intf__DOT__w_field_wen = vlSelfRef.pucb_intf__DOT__i_field_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4186, vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_consume_sel = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 76, vlSelfRef.pucb_intf__DOT__i_cb_consume_buf, vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf 
            = ((0xf0U & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 84, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf 
            = ((0xfU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_consume_buf)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf)));
    }
    vlSelfRef.pucb_intf__DOT__w_cb_consume_buf = vlSelfRef.pucb_intf__DOT__i_cb_consume_buf;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3594, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wo_field_table_vtag = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__i_cb_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_en))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 6, vlSelfRef.pucb_intf__DOT__i_cb_en, vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_en);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_cb_en 
            = vlSelfRef.pucb_intf__DOT__i_cb_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3672, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_consume = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__i_rst) ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4, vlSelfRef.pucb_intf__DOT__i_rst, vlSelfRef.pucb_intf__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__i_rst;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst 
        = vlSelfRef.pucb_intf__DOT__i_rst;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst 
        = vlSelfRef.pucb_intf__DOT__i_rst;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst 
        = vlSelfRef.pucb_intf__DOT__i_rst;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst 
        = vlSelfRef.pucb_intf__DOT__i_rst;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst 
        = vlSelfRef.pucb_intf__DOT__i_rst;
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst 
        = vlSelfRef.pucb_intf__DOT__i_rst;
    vlSelfRef.pucb_intf__DOT__w_field_stall = (1U & 
                                               ((~ (IData)(vlSelfRef.pucb_intf__DOT__i_cb_en)) 
                                                | (IData)(vlSelfRef.pucb_intf__DOT__i_rst)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3696, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__i_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 8, vlSelfRef.pucb_intf__DOT__i_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_mshr 
            = vlSelfRef.pucb_intf__DOT__i_field_mshr;
    }
    vlSelfRef.pucb_intf__DOT__w_field_mshr = vlSelfRef.pucb_intf__DOT__i_field_mshr;
    if ((vlSelfRef.pucb_intf__DOT__i_field_addr ^ vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_addr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 12, vlSelfRef.pucb_intf__DOT__i_field_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__i_field_addr 
            = vlSelfRef.pucb_intf__DOT__i_field_addr;
    }
    vlSelfRef.pucb_intf__DOT__w_field_addr = vlSelfRef.pucb_intf__DOT__i_field_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3498, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_clk);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_clk 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 882, vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset;
    if ((vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 664, vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 870, vlSelfRef.pucb_intf__DOT__wo_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__wo_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__o_cb_ptag = vlSelfRef.pucb_intf__DOT__wo_cb_ctag;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__wo_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 1189, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_clk);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_clk 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2008, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_clk);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_clk 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2620, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_clk);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_clk 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3082, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_clk);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_clk 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_clk))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4180, vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_clk);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_clk 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 440, vlSelfRef.pucb_intf__DOT__w_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_wen 
            = vlSelfRef.pucb_intf__DOT__w_field_wen;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_wen 
        = vlSelfRef.pucb_intf__DOT__w_field_wen;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen 
        = vlSelfRef.pucb_intf__DOT__w_field_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 886, vlSelfRef.pucb_intf__DOT__w_cb_consume_sel, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_sel);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_sel 
            = vlSelfRef.pucb_intf__DOT__w_cb_consume_sel;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume_sel;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume_sel;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 822, vlSelfRef.pucb_intf__DOT__w_cb_consume_buf, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf 
            = ((0xf0U & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 830, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf 
            = ((0xfU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_buf)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf)));
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume_buf;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 650, vlSelfRef.pucb_intf__DOT__wo_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 856, vlSelfRef.pucb_intf__DOT__wo_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_consume 
            = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__o_cb_consume = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 1191, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2010, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2622, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3084, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3500, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_rst))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4182, vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_rst);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_rst 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_field_stall) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_stall))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 438, vlSelfRef.pucb_intf__DOT__w_field_stall, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_stall);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_stall 
            = vlSelfRef.pucb_intf__DOT__w_field_stall;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_stall 
        = vlSelfRef.pucb_intf__DOT__w_field_stall;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall 
        = vlSelfRef.pucb_intf__DOT__w_field_stall;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall 
        = vlSelfRef.pucb_intf__DOT__w_field_stall;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3682, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 506, vlSelfRef.pucb_intf__DOT__w_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_mshr 
            = vlSelfRef.pucb_intf__DOT__w_field_mshr;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_mshr 
        = vlSelfRef.pucb_intf__DOT__w_field_mshr;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr 
        = vlSelfRef.pucb_intf__DOT__w_field_mshr;
    if ((vlSelfRef.pucb_intf__DOT__w_field_addr ^ vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_addr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 442, vlSelfRef.pucb_intf__DOT__w_field_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_addr 
            = vlSelfRef.pucb_intf__DOT__w_field_addr;
    }
    vlSelfRef.pucb_intf__DOT__w_field_atag = (vlSelfRef.pucb_intf__DOT__w_field_addr 
                                              >> 5U);
    vlSelfRef.pucb_intf__DOT__w_field_set = (0x1fU 
                                             & vlSelfRef.pucb_intf__DOT__w_field_addr);
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 170, vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset, vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset;
    }
    vlSelfRef.o_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset;
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1275, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_cb_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 162, vlSelfRef.pucb_intf__DOT__o_cb_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_ptag 
            = vlSelfRef.pucb_intf__DOT__o_cb_ptag;
    }
    vlSelfRef.o_cb_ptag = vlSelfRef.pucb_intf__DOT__o_cb_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 416, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag;
    }
    vlSelfRef.dbg_pucb_intf_wo_cb_ctag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 186, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_wen 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_wen;
    }
    vlSelfRef.dbg_pucb_intf_w_field_wen = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3834, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_wen);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_wen 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 428, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_consume_sel = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3928, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel;
    }
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3912, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru 
            = ((0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3920, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru 
            = ((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru)));
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru) 
                   >> (7U & VL_SHIFTL_III(3,3,32, (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel), 2U))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 332, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag;
    }
    vlSelfRef.dbg_pucb_intf_wo_field_table_vtag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3102, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data 
        = (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict) 
            << 7U) | (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data));
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 1261, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr) 
                    >> 2U));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line 
        = (3U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3930, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 160, vlSelfRef.pucb_intf__DOT__o_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_consume 
            = vlSelfRef.pucb_intf__DOT__o_cb_consume;
    }
    vlSelfRef.o_cb_consume = vlSelfRef.pucb_intf__DOT__o_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 402, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume;
    }
    vlSelfRef.dbg_pucb_intf_wo_cb_consume = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2048, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data 
        = (1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 1339, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3118, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag 
        = ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume) 
           | (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3832, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_stall) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_stall))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 184, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_stall, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_stall);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_stall 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_stall;
    }
    vlSelfRef.dbg_pucb_intf_w_field_stall = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_stall;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_stall))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2656, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_stall);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_stall 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_stall))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2012, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_stall);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_stall 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 888, vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr 
        = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 188, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_mshr 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_mshr;
    }
    vlSelfRef.dbg_pucb_intf_w_field_mshr = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_mshr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2014, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_mshr);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_mshr 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr;
    }
    if ((vlSelfRef.pucb_intf__DOT__w_field_atag ^ vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_atag)) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 508, vlSelfRef.pucb_intf__DOT__w_field_atag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_atag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_atag 
            = vlSelfRef.pucb_intf__DOT__w_field_atag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_atag 
        = vlSelfRef.pucb_intf__DOT__w_field_atag;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag 
        = vlSelfRef.pucb_intf__DOT__w_field_atag;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag 
        = vlSelfRef.pucb_intf__DOT__w_field_atag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_field_set) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_set))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 562, vlSelfRef.pucb_intf__DOT__w_field_set, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_set);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_set 
            = vlSelfRef.pucb_intf__DOT__w_field_set;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_set 
        = vlSelfRef.pucb_intf__DOT__w_field_set;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__w_field_set;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set 
        = vlSelfRef.pucb_intf__DOT__w_field_set;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rd_addr 
        = vlSelfRef.pucb_intf__DOT__w_field_set;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr 
        = vlSelfRef.pucb_intf__DOT__w_field_set;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr 
        = vlSelfRef.pucb_intf__DOT__w_field_set;
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1927, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 4142, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data))) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3184, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 1657, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1667, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en) 
                   << (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2092, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3200, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 430, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag;
    }
    vlSelfRef.dbg_pucb_intf_wo_cb_lru_ptag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3094, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3086, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr;
    if ((vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_atag 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_atag)) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 190, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_atag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_atag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_atag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_atag;
    }
    vlSelfRef.dbg_pucb_intf_w_field_atag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_atag;
    if ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag 
         ^ vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_atag)) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 3836, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_atag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_atag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data 
        = ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag 
            << 4U) | (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag));
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_atag)) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1193, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_atag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_atag 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_set) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_set))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 244, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_set, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_set);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_set 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_set;
    }
    vlSelfRef.dbg_pucb_intf_w_field_set = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_set;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2634, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_set))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3890, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_set);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_set 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2022, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rd_addr, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rd_addr);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rd_addr 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr 
        = (0x7fU & VL_SHIFTL_III(7,7,32, (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rd_addr), 2U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2624, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rd_addr);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rd_addr 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd 
        = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
        [vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr];
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 1251, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rd_addr);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rd_addr 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U] 
        = (IData)((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                    [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                    [1U])) << 0x20U) 
                   | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [0U]))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U] 
        = (IData)(((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [1U])) << 0x20U) 
                    | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                      [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                      [0U]))) >> 0x20U));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
        = (IData)((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                    [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                    [3U])) << 0x20U) 
                   | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [2U]))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
        = (IData)(((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [3U])) << 0x20U) 
                    | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                      [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                      [2U]))) >> 0x20U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1991, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3144, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en 
        = (0xffffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag) 
                      << (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3136, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
        [vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr];
    if ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 4078, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 2070, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_addr);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_addr 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data 
        = (((8U & ((vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                    (3U & (((IData)(3U) + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)) 
                           >> 5U))] >> (0x1fU & ((IData)(3U) 
                                                 + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)))) 
                   << 3U)) | (4U & ((vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                     (3U & (((IData)(2U) 
                                             + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)) 
                                            >> 5U))] 
                                     >> (0x1fU & ((IData)(2U) 
                                                  + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)))) 
                                    << 2U))) | ((2U 
                                                 & ((vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                                     (3U 
                                                      & (((IData)(1U) 
                                                          + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)) 
                                                         >> 5U))] 
                                                     >> 
                                                     (0x1fU 
                                                      & ((IData)(1U) 
                                                         + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)))) 
                                                    << 1U)) 
                                                | (1U 
                                                   & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                                      ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr) 
                                                       >> 5U)] 
                                                      >> 
                                                      (0x1fU 
                                                       & (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr))))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2668, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits 
        = (7U & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd) 
                 >> 2U));
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur 
        = (3U & (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd));
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[0U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1671, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[0U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[0U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U];
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[1U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1735, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[1U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[1U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U];
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[2U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1799, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[2U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[2U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U];
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[3U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1863, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[3U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[3U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U];
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag 
        = (((0xf000U & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
                        >> 0xfU)) | (0xf00U & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
                                               >> 0x13U))) 
           | ((0xf0U & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U] 
                        >> 0x17U)) | (0xfU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U] 
                                              >> 0x1bU))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
        = (IData)((((QData)((IData)((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U]))) 
                    << 0x1bU) | (QData)((IData)((0x7ffffffU 
                                                 & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U])))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
        = ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
            << 0x16U) | (IData)(((((QData)((IData)(
                                                   (0x7ffffffU 
                                                    & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U]))) 
                                   << 0x1bU) | (QData)((IData)(
                                                               (0x7ffffffU 
                                                                & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U])))) 
                                 >> 0x20U)));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
        = ((0xfffe0000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]) 
           | (0x1ffffU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
                          >> 0xaU)));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
        = ((0x1ffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]) 
           | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
              << 0x11U));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
        = (0xfffU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
                     >> 0xfU));
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en))) {
        VL_COV_TOGGLE_CHG_ST_I(16, vlSymsp->__Vcoverage + 3202, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data))) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3152, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr 
        = (0x7fU & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data));
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null 
        = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data) 
                 >> 7U));
    if ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3974, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__w_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 2084, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr)
                    ? ([&]() {
                    ++(vlSymsp->__Vcoverage[2610]);
                }(), 0xffffffffU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[2611]);
                }(), (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 2658, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 2664, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_cur = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1609, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1617, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1625, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1633, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag;
    if ((0x7ffffffU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                       ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1393, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
            = ((0xf8000000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]) 
               | (0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U]));
    }
    if ((0x7ffffffU & (((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                         << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                   >> 0x1bU)) ^ ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                                  << 5U) 
                                                 | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
                                                    >> 0x1bU))))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1447, 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                 << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                           >> 0x1bU)), 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                 << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
                                           >> 0x1bU)));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
            = ((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]) 
               | (0xf8000000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U]));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
            = ((0xffc00000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U]) 
               | (0x3fffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U]));
    }
    if ((0x7ffffffU & (((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                         << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                     >> 0x16U)) ^ (
                                                   (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                                    << 0xaU) 
                                                   | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                                      >> 0x16U))))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1501, 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                 << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                             >> 0x16U)), 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                 << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                             >> 0x16U)));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
            = ((0x3fffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U]) 
               | (0xffc00000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U]));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
            = ((0xfffe0000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U]) 
               | (0x1ffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]));
    }
    if ((0x7ffffffU & (((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                         << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                     >> 0x11U)) ^ (
                                                   (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[3U] 
                                                    << 0xfU) 
                                                   | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                                      >> 0x11U))))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1555, 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                                 << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                             >> 0x11U)), 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[3U] 
                                 << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                             >> 0x11U)));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
            = ((0x1ffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U]) 
               | (0xfffe0000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[3U] 
            = (0xfffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U]);
    }
    VL_ASSIGNBIT_II(0U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U]) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1649]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1650]);
                }(), 0U)));
    VL_ASSIGNBIT_II(1U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                      << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                                >> 0x1bU))) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1651]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1652]);
                }(), 0U)));
    VL_ASSIGNBIT_II(2U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                      << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                                  >> 0x16U))) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1653]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1654]);
                }(), 0U)));
    VL_ASSIGNBIT_II(3U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                                      << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                                  >> 0x11U))) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1655]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1656]);
                }(), 0U)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3170, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3168, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null;
    if ((vlSelfRef.pucb_intf__DOT__w_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 586, vlSelfRef.pucb_intf__DOT__w_field_table_wr_data, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__w_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__w_field_table_wr_data;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__w_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 2060, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld 
        = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 804, vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits;
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_cur) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 800, vlSelfRef.pucb_intf__DOT__w_cb_lru_cur, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1357, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1365, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1373, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1381, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__w_comp_ptag = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1641, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3122, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3120, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null;
    if ((vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3516, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data;
    }
    if ((vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 268, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data;
    }
    vlSelfRef.dbg_pucb_intf_w_field_table_wr_data = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 2052, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld;
    }
    vlSelfRef.pucb_intf__DOT__w_comp_pvld = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 366, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_cur_bits = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1131, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits 
        = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 362, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_cur = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 2016, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1028, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1247, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 752, vlSelfRef.pucb_intf__DOT__w_comp_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 760, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 768, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 776, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag 
        = vlSelfRef.pucb_intf__DOT__w_comp_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1341, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit;
    }
    vlSelfRef.pucb_intf__DOT__w_comp_hit = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 838, vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 852, vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_comp_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 792, vlSelfRef.pucb_intf__DOT__w_comp_pvld, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_pvld);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_pvld 
            = vlSelfRef.pucb_intf__DOT__w_comp_pvld;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld 
        = vlSelfRef.pucb_intf__DOT__w_comp_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1167, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv 
        = (1U | (((IData)((2U == (3U & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)))) 
                  << 2U) | (2U & ((~ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)) 
                                  << 1U))));
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv 
        = (1U | (((IData)((1U == (3U & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)))) 
                  << 2U) | (2U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits) 
                                  << 1U))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1052, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1389, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                   >> (0xfU & VL_SHIFTL_III(4,4,32, (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru), 2U))));
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 980, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 988, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 996, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1004, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_comp_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 744, vlSelfRef.pucb_intf__DOT__w_comp_hit, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_hit);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_hit 
            = vlSelfRef.pucb_intf__DOT__w_comp_hit;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit 
        = vlSelfRef.pucb_intf__DOT__w_comp_hit;
    vlSelfRef.pucb_intf__DOT__w_cb_miss = (1U & (~ 
                                                 (0U 
                                                  != 
                                                  ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_hit) 
                                                   & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_pvld)))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 384, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_rev_ptr = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3946, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag 
        = ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[4178]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[4179]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 398, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_rev_ptr_null = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3944, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null 
        = ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[4176]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[4177]);
            }(), 0U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1020, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1161, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits 
        = ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv) 
           ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits));
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1149, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits 
        = ((4U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv) 
                  & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits))) 
           | ((2U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv) 
                     & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits))) 
              | (1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1349, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1080, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1088, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1096, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1104, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1012, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 820, vlSelfRef.pucb_intf__DOT__w_cb_miss, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_miss);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_miss 
            = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 4152, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4150, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1072, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1173, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits 
        = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1155, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread 
        = (1U & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits));
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread 
        = (3U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits) 
                 >> 1U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 784, vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_lru_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1064, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 382, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_miss = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2654, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld 
        = ((~ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall)) 
           & (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3830, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen 
        = ((~ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss)) 
           & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2020, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume 
        = ((~ ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data) 
               >> (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru))) 
           & ((~ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall)) 
              & (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 4040, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4038, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null;
    }
    vlSelfRef.pucb_intf__DOT__wo_validity_table_null 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1137, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 1183, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1185, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt 
        = (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread) 
            << 1U) | (0U != (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1032, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2752, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen 
        = ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld) 
           << (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4166, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2068, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume 
        = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 728, vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_validity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 742, vlSelfRef.pucb_intf__DOT__wo_validity_table_null, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_null 
            = vlSelfRef.pucb_intf__DOT__wo_validity_table_null;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_null;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 814, vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1179, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt 
        = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1056, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag;
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru;
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 0U;
    if ((1U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 0U;
        ++(vlSymsp->__Vcoverage[1124]);
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 1U;
    if ((2U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        ++(vlSymsp->__Vcoverage[1124]);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                       >> 4U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 1U;
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 2U;
    if ((4U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        ++(vlSymsp->__Vcoverage[1124]);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                       >> 8U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 2U;
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 3U;
    if ((8U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        ++(vlSymsp->__Vcoverage[1124]);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                       >> 0xcU));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 3U;
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 4U;
    if ((1U & (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
                & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)) 
               >> (3U & vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k)))) {
        ++(vlSymsp->__Vcoverage[1126]);
    }
    if ((1U & (~ ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit) 
                  >> (3U & vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k))))) {
        ++(vlSymsp->__Vcoverage[1127]);
    }
    if ((1U & (~ ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
                  >> (3U & vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k))))) {
        ++(vlSymsp->__Vcoverage[1128]);
    }
    ++(vlSymsp->__Vcoverage[1130]);
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen 
         ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 2688, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4054, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen;
    }
    vlSelfRef.pucb_intf__DOT__w_mem_wen = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2050, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_consume = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 346, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag;
    }
    vlSelfRef.dbg_pucb_intf_wo_validity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 2032, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 360, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null;
    }
    vlSelfRef.dbg_pucb_intf_wo_validity_table_null 
        = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2046, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null 
        = ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall) 
           | (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null));
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 376, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_nxt_bits = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 2644, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1143, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1116, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1112, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 896, vlSelfRef.pucb_intf__DOT__w_mem_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_wen 
            = vlSelfRef.pucb_intf__DOT__w_mem_wen;
    }
    vlSelfRef.pucb_intf__DOT__o_mem_wen = vlSelfRef.pucb_intf__DOT__w_mem_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 854, vlSelfRef.pucb_intf__DOT__w_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume 
            = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2350, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null;
    }
    __Vtemp_3[0U] = 0U;
    __Vtemp_3[1U] = 0U;
    __Vtemp_3[2U] = 0U;
    __Vtemp_3[3U] = 0U;
    __Vtemp_4[0U] = 1U;
    __Vtemp_4[1U] = 0U;
    __Vtemp_4[2U] = 0U;
    __Vtemp_4[3U] = 0U;
    VL_SHIFTL_WWI(128,128,7, __Vtemp_5, __Vtemp_4, (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr));
    VL_COND_WIWW(128, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en, (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null), 
                 ([&]() {
                ++(vlSymsp->__Vcoverage[2609]);
            }(), __Vtemp_3), ([&]() {
                ++(vlSymsp->__Vcoverage[2608]);
            }(), __Vtemp_5));
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 810, vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1044, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_ctag = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1040, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_cline = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 174, vlSelfRef.pucb_intf__DOT__o_mem_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_wen 
            = vlSelfRef.pucb_intf__DOT__o_mem_wen;
    }
    vlSelfRef.o_mem_wen = vlSelfRef.pucb_intf__DOT__o_mem_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3580, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 400, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_consume = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4184, vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel 
        = (1U & (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume) 
                  + (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel)) 
                 - (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap)));
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset 
        = (3U & ((((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset) 
                   + ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel) 
                      + (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume))) 
                  - (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap)) 
                 - (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap)));
    __Vtemp_6[0U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[0U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[0U]);
    __Vtemp_6[1U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[1U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[1U]);
    __Vtemp_6[2U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[2U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[2U]);
    __Vtemp_6[3U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[3U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[3U]);
    if (__Vtemp_6) {
        VL_COV_TOGGLE_CHG_ST_W(128, vlSymsp->__Vcoverage + 2094, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[0U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[0U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[1U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[1U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[2U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[2U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[3U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[3U];
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 372, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_nxt = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 2650, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr 
        = (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits) 
            << 2U) | (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt));
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 862, vlSelfRef.pucb_intf__DOT__w_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_cline) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_cline))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 858, vlSelfRef.pucb_intf__DOT__w_cb_cline, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_cline);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_cline 
            = vlSelfRef.pucb_intf__DOT__w_cb_cline;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline 
        = vlSelfRef.pucb_intf__DOT__w_cb_cline;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline 
        = vlSelfRef.pucb_intf__DOT__w_cb_cline;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4192, vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4194, vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset 
        = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2678, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3582, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 408, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_ctag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3900, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 404, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_cline = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3908, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag 
        = (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set) 
            << 2U) | (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline));
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4188, vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 4168, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 4064, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 878, vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset 
        = vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset 
        = vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 4056, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr;
    }
    vlSelfRef.pucb_intf__DOT__w_mem_addr = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3960, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__w_field_table_vtag = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3590, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 424, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 898, vlSelfRef.pucb_intf__DOT__w_mem_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_addr 
            = vlSelfRef.pucb_intf__DOT__w_mem_addr;
    }
    vlSelfRef.pucb_intf__DOT__o_mem_addr = vlSelfRef.pucb_intf__DOT__w_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 572, vlSelfRef.pucb_intf__DOT__w_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__w_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__w_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__w_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 176, vlSelfRef.pucb_intf__DOT__o_mem_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_addr 
            = vlSelfRef.pucb_intf__DOT__o_mem_addr;
    }
    vlSelfRef.o_mem_addr = vlSelfRef.pucb_intf__DOT__o_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3502, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 254, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag;
    }
    vlSelfRef.dbg_pucb_intf_w_field_table_vtag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag;
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop___024root___eval_triggers__act(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_triggers__act\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.setBit(0U, ((IData)(vlSelfRef.pucb_intf__DOT__i_pu_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__i_pu_clk__0))));
    vlSelfRef.__VactTriggered.setBit(1U, ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__field_table_i__DOT__i_clk__0))));
    vlSelfRef.__VactTriggered.setBit(2U, ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__validity_table_i__DOT__i_clk__0))));
    vlSelfRef.__VactTriggered.setBit(3U, ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__lru_regfile_i__DOT__i_clk__0))));
    vlSelfRef.__VactTriggered.setBit(4U, ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk__0))));
    vlSelfRef.__VactTriggered.setBit(5U, ((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__in_cdc_i__DOT__i_clk__0))));
    vlSelfRef.__VactTriggered.setBit(6U, ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__controller_i__DOT__i_clk__0))));
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__i_pu_clk__0 
        = vlSelfRef.pucb_intf__DOT__i_pu_clk;
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__field_table_i__DOT__i_clk__0 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk;
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__validity_table_i__DOT__i_clk__0 
        = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk;
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__lru_regfile_i__DOT__i_clk__0 
        = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk;
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk__0 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk;
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__in_cdc_i__DOT__i_clk__0 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk;
    vlSelfRef.__Vtrigprevexpr___TOP__pucb_intf__DOT__controller_i__DOT__i_clk__0 
        = vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop___024root___dump_triggers__act(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__0(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__0\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr 
        = (3U & (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst) 
                  | (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en))
                  ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3805]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3806]);
                }(), ((IData)(1U) + (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr)))));
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) {
        ++(vlSymsp->__Vcoverage[3802]);
    }
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[3803]);
    }
    if ((1U & ((~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst)) 
               & (~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en))))) {
        ++(vlSymsp->__Vcoverage[3804]);
    }
    ++(vlSymsp->__Vcoverage[3807]);
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag 
        = ((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[3811]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[3812]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag)));
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data 
        = ((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[3815]);
            }(), vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data)
            : ([&]() {
                ++(vlSymsp->__Vcoverage[3816]);
            }(), vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data));
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume 
        = ((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[3819]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[3820]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume)));
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag 
        = ((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[3823]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[3824]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag)));
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset 
        = ((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[3827]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[3828]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset)));
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) {
        ++(vlSymsp->__Vcoverage[3809]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)))) {
        ++(vlSymsp->__Vcoverage[3810]);
    }
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) {
        ++(vlSymsp->__Vcoverage[3813]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)))) {
        ++(vlSymsp->__Vcoverage[3814]);
    }
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) {
        ++(vlSymsp->__Vcoverage[3817]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)))) {
        ++(vlSymsp->__Vcoverage[3818]);
    }
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) {
        ++(vlSymsp->__Vcoverage[3821]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)))) {
        ++(vlSymsp->__Vcoverage[3822]);
    }
    if (vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) {
        ++(vlSymsp->__Vcoverage[3825]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en)))) {
        ++(vlSymsp->__Vcoverage[3826]);
    }
    ++(vlSymsp->__Vcoverage[3829]);
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3704, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en = 
        ((1U == (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr))
          ? ([&]() {
                ++(vlSymsp->__Vcoverage[3800]);
            }(), 1U) : ([&]() {
                ++(vlSymsp->__Vcoverage[3801]);
            }(), 0U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3796, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3788, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3708, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3786, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume;
    if ((vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3722, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data;
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag 
        = (0xfU & vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data);
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3694, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3690, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3674, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_ctag = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3594, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wo_field_table_vtag = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3672, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_consume = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume;
    if ((vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3608, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3696, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 882, vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 870, vlSelfRef.pucb_intf__DOT__wo_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__wo_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__o_cb_ptag = vlSelfRef.pucb_intf__DOT__wo_cb_ctag;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__wo_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 650, vlSelfRef.pucb_intf__DOT__wo_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 856, vlSelfRef.pucb_intf__DOT__wo_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_consume 
            = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__o_cb_consume = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume 
        = vlSelfRef.pucb_intf__DOT__wo_cb_consume;
    if ((vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 664, vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data 
        = vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3682, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 170, vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset, vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset;
    }
    vlSelfRef.o_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_cb_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 162, vlSelfRef.pucb_intf__DOT__o_cb_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_ptag 
            = vlSelfRef.pucb_intf__DOT__o_cb_ptag;
    }
    vlSelfRef.o_cb_ptag = vlSelfRef.pucb_intf__DOT__o_cb_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 416, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag;
    }
    vlSelfRef.dbg_pucb_intf_wo_cb_ctag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 332, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag;
    }
    vlSelfRef.dbg_pucb_intf_wo_field_table_vtag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3102, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 1261, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line 
        = (3U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3930, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 160, vlSelfRef.pucb_intf__DOT__o_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_cb_consume 
            = vlSelfRef.pucb_intf__DOT__o_cb_consume;
    }
    vlSelfRef.o_cb_consume = vlSelfRef.pucb_intf__DOT__o_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 402, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume;
    }
    vlSelfRef.dbg_pucb_intf_wo_cb_consume = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2048, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 1339, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3118, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag 
        = ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume) 
           | (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3832, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume;
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1275, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 888, vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr 
        = vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1667, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3200, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 430, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag;
    }
    vlSelfRef.dbg_pucb_intf_wo_cb_lru_ptag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3094, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3086, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3144, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3136, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__1(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__1\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg 
        = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)
                  ? ([&]() {
                    ++(vlSymsp->__Vcoverage[4216]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[4217]);
                }(), (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume))));
    if (vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[4214]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[4215]);
    }
    ++(vlSymsp->__Vcoverage[4218]);
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg 
        = (3U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)
                  ? ([&]() {
                    ++(vlSymsp->__Vcoverage[4221]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[4222]);
                }(), ((2U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg) 
                             << 1U)) | (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume)))));
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel 
        = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)
                  ? ([&]() {
                    ++(vlSymsp->__Vcoverage[4225]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[4226]);
                }(), (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel))));
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset 
        = (3U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)
                  ? ([&]() {
                    ++(vlSymsp->__Vcoverage[4229]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[4230]);
                }(), (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset))));
    if (vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[4219]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[4220]);
    }
    if (vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[4223]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[4224]);
    }
    if (vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[4227]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[4228]);
    }
    ++(vlSymsp->__Vcoverage[4231]);
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4202, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg;
    }
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap 
        = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4210, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4204, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg;
    }
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap 
        = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg) 
                 >> 1U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4208, vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel;
    }
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel 
        = vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4198, vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4200, vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4186, vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_consume_sel = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 886, vlSelfRef.pucb_intf__DOT__w_cb_consume_sel, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_sel);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume_sel 
            = vlSelfRef.pucb_intf__DOT__w_cb_consume_sel;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume_sel;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume_sel;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 428, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_consume_sel = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3928, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru) 
                   >> (7U & VL_SHIFTL_III(3,3,32, (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel), 2U))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 4142, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data 
        = ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag 
            << 4U) | (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag));
    if ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 4078, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data;
    if ((vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3974, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__w_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data;
    if ((vlSelfRef.pucb_intf__DOT__w_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 586, vlSelfRef.pucb_intf__DOT__w_field_table_wr_data, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__w_field_table_wr_data;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__w_field_table_wr_data;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data 
        = vlSelfRef.pucb_intf__DOT__w_field_table_wr_data;
    if ((vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 3516, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data;
    }
    if ((vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data 
         ^ vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 268, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data;
    }
    vlSelfRef.dbg_pucb_intf_w_field_table_wr_data = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data;
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__2(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__2\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*31:0*/ __Vtemp_1;
    VlWide<4>/*127:0*/ __Vtemp_2;
    // Body
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k = 0U;
    while (VL_GTS_III(32, 0x80U, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k)) {
        __Vtemp_1 = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst)
                            ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2614]);
                    }(), 0U) : ([&]() {
                        ++(vlSymsp->__Vcoverage[2617]);
                    }(), ((1U & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[
                                 (3U & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k 
                                        >> 5U))] >> 
                                 (0x1fU & vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k)))
                           ? ([&]() {
                                ++(vlSymsp->__Vcoverage[2615]);
                            }(), (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data))
                           : ([&]() {
                                ++(vlSymsp->__Vcoverage[2616]);
                            }(), (1U & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                        (3U & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k 
                                               >> 5U))] 
                                        >> (0x1fU & vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k))))))));
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[(3U 
                                                                      & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k 
                                                                         >> 5U))] 
            = (((~ ((IData)(1U) << (0x1fU & vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k))) 
                & vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                (3U & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k 
                       >> 5U))]) | (__Vtemp_1 << (0x1fU 
                                                  & vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k)));
        ++(vlSymsp->__Vcoverage[2618]);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k 
            = ((IData)(1U) + vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k);
    }
    if (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[2612]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[2613]);
    }
    ++(vlSymsp->__Vcoverage[2619]);
    __Vtemp_2[0U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[0U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[0U]);
    __Vtemp_2[1U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[1U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[1U]);
    __Vtemp_2[2U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[2U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[2U]);
    __Vtemp_2[3U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[3U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[3U]);
    if (__Vtemp_2) {
        VL_COV_TOGGLE_CHG_ST_W(128, vlSymsp->__Vcoverage + 2352, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[0U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[0U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[1U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[1U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[2U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[2U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table[3U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[3U];
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data 
        = (((8U & ((vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                    (3U & (((IData)(3U) + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)) 
                           >> 5U))] >> (0x1fU & ((IData)(3U) 
                                                 + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)))) 
                   << 3U)) | (4U & ((vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                     (3U & (((IData)(2U) 
                                             + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)) 
                                            >> 5U))] 
                                     >> (0x1fU & ((IData)(2U) 
                                                  + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)))) 
                                    << 2U))) | ((2U 
                                                 & ((vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                                     (3U 
                                                      & (((IData)(1U) 
                                                          + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)) 
                                                         >> 5U))] 
                                                     >> 
                                                     (0x1fU 
                                                      & ((IData)(1U) 
                                                         + (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr)))) 
                                                    << 1U)) 
                                                | (1U 
                                                   & (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table[
                                                      ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr) 
                                                       >> 5U)] 
                                                      >> 
                                                      (0x1fU 
                                                       & (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr))))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 2084, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr)
                    ? ([&]() {
                    ++(vlSymsp->__Vcoverage[2610]);
                }(), 0xffffffffU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[2611]);
                }(), (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 2060, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld 
        = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 2052, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld;
    }
    vlSelfRef.pucb_intf__DOT__w_comp_pvld = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_comp_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 792, vlSelfRef.pucb_intf__DOT__w_comp_pvld, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_pvld);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_pvld 
            = vlSelfRef.pucb_intf__DOT__w_comp_pvld;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld 
        = vlSelfRef.pucb_intf__DOT__w_comp_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1020, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1072, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__3(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__3\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v0;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v0 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v1;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v1 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v2;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v2 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v3;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v3 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v4;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v4 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v5;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v5 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v6;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v6 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v7;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v7 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v8;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v8 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v9;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v9 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v10;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v10 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v11;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v11 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v12;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v12 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v13;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v13 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v14;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v14 = 0;
    CData/*7:0*/ __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v15;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v15 = 0;
    // Body
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 1U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 2U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 3U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 4U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 5U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 6U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 7U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 8U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 9U;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0xaU;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0xbU;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0xcU;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0xdU;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0xeU;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0xfU;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k = 0x10U;
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v0 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x80U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((1U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v1 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x81U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((2U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [1U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v2 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x82U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((4U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [2U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v3 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x83U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((8U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [3U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v4 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x84U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x10U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [4U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v5 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x85U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x20U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [5U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v6 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x86U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x40U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [6U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v7 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x87U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x80U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [7U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v8 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x88U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x100U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [8U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v9 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x89U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x200U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [9U])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v10 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x8aU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x400U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0xaU])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v11 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x8bU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x800U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0xbU])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v12 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x8cU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x1000U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0xcU])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v13 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x8dU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x2000U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0xdU])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v14 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x8eU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x4000U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0xeU])))));
    ++(vlSymsp->__Vcoverage[3496]);
    __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v15 
        = (0xffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3492]);
                }(), 0x8fU) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3495]);
                }(), ((0x8000U & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en))
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3493]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3494]);
                        }(), vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                          [0xfU])))));
    ++(vlSymsp->__Vcoverage[3496]);
    if (vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[3490]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[3491]);
    }
    ++(vlSymsp->__Vcoverage[3497]);
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v0;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[1U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v1;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[2U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v2;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[3U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v3;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[4U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v4;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[5U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v5;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[6U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v6;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[7U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v7;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[8U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v8;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[9U] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v9;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0xaU] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v10;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0xbU] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v11;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0xcU] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v12;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0xdU] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v13;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0xeU] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v14;
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0xfU] 
        = __VdlyVal__pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table__v15;
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3234, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [1U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [1U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3250, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [1U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [1U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[1U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [1U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [2U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [2U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3266, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [2U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [2U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[2U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [2U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [3U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [3U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3282, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [3U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [3U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[3U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [3U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [4U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [4U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3298, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [4U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [4U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[4U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [4U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [5U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [5U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3314, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [5U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [5U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[5U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [5U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [6U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [6U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3330, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [6U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [6U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[6U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [6U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [7U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [7U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3346, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [7U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [7U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[7U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [7U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [8U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [8U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3362, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [8U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [8U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[8U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [8U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [9U] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [9U])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3378, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [9U], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [9U]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[9U] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [9U];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xaU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xaU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3394, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xaU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xaU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xaU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xaU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xbU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xbU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3410, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xbU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xbU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xbU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xbU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xcU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xcU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3426, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xcU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xcU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xcU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xcU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xdU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xdU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3442, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xdU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xdU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xdU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xdU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xeU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xeU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3458, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xeU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xeU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xeU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xeU];
    }
    if ((vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
         [0xfU] ^ vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
         [0xfU])) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3474, 
                               vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
                               [0xfU], vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table
                               [0xfU]);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[0xfU] 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
            [0xfU];
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__4(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__4\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v0;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v0 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v1;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v1 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v2;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v2 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v3;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v3 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v4;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v4 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v5;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v5 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v6;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v6 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v7;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v7 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v8;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v8 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v9;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v9 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v10;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v10 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v11;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v11 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v12;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v12 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v13;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v13 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v14;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v14 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v15;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v15 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v16;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v16 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v17;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v17 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v18;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v18 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v19;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v19 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v20;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v20 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v21;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v21 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v22;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v22 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v23;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v23 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v24;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v24 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v25;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v25 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v26;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v26 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v27;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v27 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v28;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v28 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v29;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v29 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v30;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v30 = 0;
    CData/*4:0*/ __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v31;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v31 = 0;
    // Body
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 1U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 2U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 3U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 4U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 5U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 6U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 7U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 8U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 9U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0xaU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0xbU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0xcU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0xdU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0xeU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0xfU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x10U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x11U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x12U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x13U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x14U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x15U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x16U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x17U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x18U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x19U;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x1aU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x1bU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x1cU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x1dU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x1eU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x1fU;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k = 0x20U;
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v0 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((1U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v1 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((2U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [1U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v2 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((4U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [2U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v3 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((8U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [3U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v4 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x10U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [4U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v5 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x20U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [5U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v6 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x40U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [6U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v7 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x80U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [7U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v8 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x100U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [8U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v9 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x200U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [9U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v10 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x400U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0xaU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v11 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x800U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0xbU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v12 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x1000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0xcU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v13 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x2000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0xdU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v14 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x4000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0xeU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v15 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x8000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0xfU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v16 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x10000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x10U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v17 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x20000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x11U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v18 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x40000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x12U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v19 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x80000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x13U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v20 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x100000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x14U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v21 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x200000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x15U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v22 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x400000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x16U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v23 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x800000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x17U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v24 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x1000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x18U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v25 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x2000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x19U])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v26 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x4000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x1aU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v27 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x8000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x1bU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v28 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x10000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x1cU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v29 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x20000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x1dU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v30 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((0x40000000U & vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen)
                       ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x1eU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v31 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)
                     ? ([&]() {
                    ++(vlSymsp->__Vcoverage[3076]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[3079]);
                }(), ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen 
                       >> 0x1fU) ? ([&]() {
                            ++(vlSymsp->__Vcoverage[3077]);
                        }(), (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr))
                       : ([&]() {
                            ++(vlSymsp->__Vcoverage[3078]);
                        }(), vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                          [0x1fU])))));
    ++(vlSymsp->__Vcoverage[3080]);
    if (vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[3074]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[3075]);
    }
    ++(vlSymsp->__Vcoverage[3081]);
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v0;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[1U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v1;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[2U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v2;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[3U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v3;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[4U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v4;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[5U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v5;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[6U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v6;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[7U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v7;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[8U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v8;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[9U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v9;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0xaU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v10;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0xbU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v11;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0xcU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v12;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0xdU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v13;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0xeU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v14;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0xfU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v15;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x10U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v16;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x11U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v17;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x12U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v18;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x13U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v19;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x14U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v20;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x15U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v21;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x16U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v22;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x17U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v23;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x18U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v24;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x19U] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v25;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x1aU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v26;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x1bU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v27;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x1cU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v28;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x1dU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v29;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x1eU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v30;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0x1fU] 
        = __VdlyVal__pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state__v31;
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2754, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [1U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [1U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2764, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [1U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [1U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[1U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [1U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [2U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [2U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2774, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [2U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [2U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[2U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [2U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [3U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [3U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2784, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [3U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [3U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[3U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [3U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [4U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [4U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2794, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [4U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [4U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[4U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [4U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [5U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [5U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2804, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [5U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [5U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[5U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [5U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [6U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [6U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2814, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [6U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [6U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[6U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [6U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [7U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [7U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2824, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [7U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [7U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[7U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [7U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [8U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [8U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2834, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [8U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [8U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[8U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [8U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [9U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [9U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2844, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [9U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [9U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[9U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [9U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xaU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xaU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2854, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xaU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xaU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xaU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xaU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xbU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xbU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2864, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xbU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xbU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xbU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xbU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xcU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xcU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2874, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xcU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xcU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xcU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xcU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xdU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xdU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2884, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xdU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xdU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xdU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xdU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xeU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xeU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2894, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xeU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xeU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xeU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xeU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0xfU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0xfU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2904, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0xfU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0xfU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0xfU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0xfU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x10U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x10U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2914, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x10U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x10U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x10U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x10U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x11U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x11U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2924, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x11U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x11U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x11U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x11U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x12U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x12U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2934, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x12U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x12U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x12U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x12U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x13U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x13U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2944, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x13U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x13U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x13U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x13U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x14U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x14U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2954, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x14U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x14U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x14U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x14U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x15U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x15U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2964, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x15U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x15U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x15U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x15U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x16U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x16U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2974, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x16U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x16U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x16U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x16U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x17U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x17U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2984, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x17U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x17U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x17U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x17U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x18U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x18U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2994, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x18U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x18U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x18U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x18U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x19U] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x19U])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3004, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x19U], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x19U]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x19U] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x19U];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1aU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1aU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3014, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1aU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1aU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1aU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1aU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1bU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1bU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3024, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1bU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1bU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1bU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1bU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1cU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1cU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3034, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1cU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1cU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1cU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1cU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1dU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1dU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3044, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1dU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1dU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1dU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1dU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1eU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1eU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3054, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1eU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1eU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1eU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1eU];
    }
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
         [0x1fU] ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
         [0x1fU])) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 3064, 
                               vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
                               [0x1fU], vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state
                               [0x1fU]);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[0x1fU] 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
            [0x1fU];
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd 
        = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state
        [vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr];
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2668, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits 
        = (7U & ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd) 
                 >> 2U));
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur 
        = (3U & (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd));
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 2658, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 2664, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_cur = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 804, vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits;
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_cur) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 800, vlSelfRef.pucb_intf__DOT__w_cb_lru_cur, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_cur 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_cur;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 366, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_cur_bits = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1131, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits 
        = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 362, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_cur = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 2016, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1028, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1247, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1167, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv 
        = (1U | (((IData)((2U == (3U & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)))) 
                  << 2U) | (2U & ((~ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)) 
                                  << 1U))));
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv 
        = (1U | (((IData)((1U == (3U & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)))) 
                  << 2U) | (2U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits) 
                                  << 1U))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1052, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1389, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1161, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits 
        = ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv) 
           ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits));
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1149, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits 
        = ((4U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv) 
                  & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits))) 
           | ((2U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv) 
                     & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits))) 
              | (1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits)))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1173, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits 
        = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1155, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread 
        = (1U & (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits));
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread 
        = (3U & ((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits) 
                 >> 1U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 1137, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 1183, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1185, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt 
        = (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread) 
            << 1U) | (0U != (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 814, vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1179, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt 
        = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 376, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_nxt_bits = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits))) {
        VL_COV_TOGGLE_CHG_ST_I(3, vlSymsp->__Vcoverage + 2644, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1143, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt, vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt);
        vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt = vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 810, vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt 
        = vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 372, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_lru_nxt = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 2650, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr 
        = (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits) 
            << 2U) | (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt));
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 2678, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__5(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__5\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v0;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v0 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v1;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v1 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v2;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v2 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v3;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v3 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v4;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v4 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v5;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v5 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v6;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v6 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v7;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v7 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v8;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v8 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v9;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v9 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v10;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v10 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v11;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v11 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v12;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v12 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v13;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v13 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v14;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v14 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v15;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v15 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v16;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v16 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v17;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v17 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v18;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v18 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v19;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v19 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v20;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v20 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v21;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v21 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v22;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v22 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v23;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v23 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v24;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v24 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v25;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v25 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v26;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v26 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v27;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v27 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v28;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v28 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v29;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v29 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v30;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v30 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v31;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v31 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v32;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v32 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v33;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v33 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v34;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v34 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v35;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v35 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v36;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v36 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v37;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v37 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v38;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v38 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v39;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v39 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v40;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v40 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v41;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v41 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v42;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v42 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v43;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v43 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v44;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v44 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v45;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v45 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v46;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v46 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v47;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v47 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v48;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v48 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v49;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v49 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v50;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v50 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v51;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v51 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v52;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v52 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v53;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v53 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v54;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v54 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v55;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v55 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v56;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v56 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v57;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v57 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v58;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v58 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v59;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v59 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v60;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v60 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v61;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v61 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v62;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v62 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v63;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v63 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v64;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v64 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v65;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v65 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v66;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v66 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v67;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v67 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v68;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v68 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v69;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v69 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v70;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v70 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v71;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v71 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v72;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v72 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v73;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v73 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v74;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v74 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v75;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v75 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v76;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v76 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v77;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v77 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v78;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v78 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v79;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v79 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v80;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v80 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v81;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v81 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v82;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v82 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v83;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v83 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v84;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v84 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v85;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v85 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v86;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v86 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v87;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v87 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v88;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v88 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v89;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v89 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v90;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v90 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v91;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v91 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v92;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v92 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v93;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v93 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v94;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v94 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v95;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v95 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v96;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v96 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v97;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v97 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v98;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v98 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v99;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v99 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v100;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v100 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v101;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v101 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v102;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v102 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v103;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v103 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v104;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v104 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v105;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v105 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v106;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v106 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v107;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v107 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v108;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v108 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v109;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v109 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v110;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v110 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v111;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v111 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v112;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v112 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v113;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v113 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v114;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v114 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v115;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v115 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v116;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v116 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v117;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v117 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v118;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v118 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v119;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v119 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v120;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v120 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v121;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v121 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v122;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v122 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v123;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v123 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v124;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v124 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v125;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v125 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v126;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v126 = 0;
    IData/*31:0*/ __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v127;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v127 = 0;
    // Body
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 5U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 6U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 7U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 8U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 9U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0xaU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0xbU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0xcU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0xdU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0xeU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0xfU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x10U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x11U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x12U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x13U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x14U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x15U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x16U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x17U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x18U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x19U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x1aU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x1bU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x1cU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x1dU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x1eU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x1fU;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k = 0x20U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 0U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 1U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 2U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 3U;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m = 4U;
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v0 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v1 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v2 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v3 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v4 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (1U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [1U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v5 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (1U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [1U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v6 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (1U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [1U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v7 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (1U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [1U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v8 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (2U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [2U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v9 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (2U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [2U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v10 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (2U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [2U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v11 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (2U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [2U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v12 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (3U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [3U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v13 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (3U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [3U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v14 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (3U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [3U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v15 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (3U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [3U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v16 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (4U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [4U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v17 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (4U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [4U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v18 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (4U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [4U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v19 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (4U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [4U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v20 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (5U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [5U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v21 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (5U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [5U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v22 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (5U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [5U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v23 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (5U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [5U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v24 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (6U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [6U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v25 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (6U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [6U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v26 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (6U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [6U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v27 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (6U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [6U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v28 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (7U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [7U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v29 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (7U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [7U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v30 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (7U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [7U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v31 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (7U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [7U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v32 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (8U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [8U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v33 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (8U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [8U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v34 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (8U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [8U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v35 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (8U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [8U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v36 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (9U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [9U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v37 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (9U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [9U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v38 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (9U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [9U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v39 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (9U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [9U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v40 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0xaU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xaU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v41 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0xaU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xaU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v42 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0xaU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xaU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v43 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0xaU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xaU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v44 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0xbU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xbU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v45 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0xbU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xbU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v46 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0xbU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xbU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v47 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0xbU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xbU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v48 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0xcU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xcU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v49 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0xcU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xcU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v50 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0xcU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xcU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v51 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0xcU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xcU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v52 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0xdU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xdU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v53 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0xdU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xdU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v54 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0xdU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xdU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v55 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0xdU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xdU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v56 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0xeU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xeU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v57 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0xeU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xeU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v58 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0xeU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xeU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v59 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0xeU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xeU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v60 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0xfU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xfU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v61 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0xfU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xfU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v62 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0xfU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xfU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v63 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0xfU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0xfU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v64 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x10U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x10U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v65 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x10U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x10U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v66 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x10U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x10U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v67 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x10U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x10U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v68 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x11U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x11U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v69 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x11U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x11U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v70 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x11U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x11U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v71 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x11U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x11U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v72 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x12U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x12U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v73 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x12U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x12U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v74 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x12U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x12U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v75 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x12U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x12U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v76 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x13U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x13U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v77 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x13U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x13U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v78 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x13U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x13U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v79 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x13U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x13U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v80 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x14U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x14U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v81 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x14U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x14U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v82 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x14U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x14U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v83 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x14U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x14U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v84 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x15U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x15U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v85 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x15U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x15U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v86 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x15U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x15U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v87 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x15U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x15U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v88 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x16U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x16U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v89 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x16U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x16U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v90 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x16U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x16U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v91 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x16U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x16U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v92 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x17U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x17U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v93 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x17U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x17U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v94 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x17U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x17U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v95 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x17U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x17U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v96 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x18U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x18U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v97 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x18U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x18U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v98 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x18U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x18U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v99 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x18U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x18U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v100 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x19U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x19U][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v101 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x19U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x19U][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v102 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x19U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x19U][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v103 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x19U == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x19U][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v104 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x1aU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1aU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v105 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x1aU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1aU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v106 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x1aU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1aU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v107 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x1aU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1aU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v108 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x1bU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1bU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v109 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x1bU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1bU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v110 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x1bU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1bU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v111 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x1bU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1bU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v112 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x1cU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1cU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v113 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x1cU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1cU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v114 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x1cU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1cU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v115 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x1cU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1cU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v116 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x1dU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1dU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v117 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x1dU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1dU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v118 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x1dU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1dU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v119 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x1dU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1dU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v120 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x1eU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1eU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v121 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x1eU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1eU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v122 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x1eU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1eU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v123 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x1eU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1eU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v124 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                   & (0x1fU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1fU][0U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v125 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 1U) & (0x1fU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1fU][1U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v126 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 2U) & (0x1fU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1fU][2U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v127 
        = ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[2001]);
            }(), 0U) : ([&]() {
                ++(vlSymsp->__Vcoverage[2004]);
            }(), ((((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
                    >> 3U) & (0x1fU == (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set)))
                   ? ([&]() {
                        ++(vlSymsp->__Vcoverage[2002]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr)
                   : ([&]() {
                        ++(vlSymsp->__Vcoverage[2003]);
                    }(), vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                      [0x1fU][3U]))));
    ++(vlSymsp->__Vcoverage[2005]);
    ++(vlSymsp->__Vcoverage[2006]);
    if (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst) {
        ++(vlSymsp->__Vcoverage[1999]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst)))) {
        ++(vlSymsp->__Vcoverage[2000]);
    }
    ++(vlSymsp->__Vcoverage[2007]);
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v0;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v1;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v2;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v3;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[1U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v4;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[1U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v5;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[1U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v6;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[1U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v7;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[2U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v8;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[2U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v9;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[2U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v10;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[2U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v11;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[3U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v12;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[3U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v13;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[3U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v14;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[3U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v15;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[4U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v16;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[4U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v17;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[4U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v18;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[4U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v19;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[5U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v20;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[5U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v21;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[5U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v22;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[5U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v23;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[6U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v24;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[6U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v25;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[6U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v26;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[6U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v27;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[7U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v28;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[7U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v29;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[7U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v30;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[7U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v31;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[8U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v32;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[8U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v33;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[8U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v34;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[8U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v35;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[9U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v36;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[9U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v37;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[9U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v38;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[9U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v39;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xaU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v40;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xaU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v41;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xaU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v42;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xaU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v43;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xbU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v44;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xbU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v45;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xbU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v46;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xbU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v47;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xcU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v48;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xcU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v49;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xcU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v50;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xcU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v51;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xdU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v52;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xdU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v53;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xdU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v54;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xdU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v55;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xeU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v56;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xeU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v57;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xeU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v58;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xeU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v59;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xfU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v60;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xfU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v61;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xfU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v62;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0xfU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v63;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x10U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v64;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x10U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v65;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x10U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v66;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x10U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v67;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x11U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v68;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x11U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v69;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x11U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v70;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x11U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v71;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x12U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v72;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x12U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v73;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x12U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v74;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x12U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v75;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x13U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v76;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x13U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v77;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x13U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v78;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x13U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v79;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x14U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v80;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x14U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v81;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x14U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v82;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x14U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v83;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x15U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v84;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x15U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v85;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x15U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v86;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x15U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v87;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x16U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v88;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x16U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v89;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x16U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v90;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x16U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v91;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x17U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v92;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x17U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v93;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x17U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v94;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x17U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v95;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x18U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v96;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x18U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v97;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x18U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v98;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x18U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v99;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x19U][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v100;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x19U][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v101;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x19U][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v102;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x19U][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v103;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1aU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v104;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1aU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v105;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1aU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v106;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1aU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v107;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1bU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v108;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1bU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v109;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1bU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v110;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1bU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v111;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1cU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v112;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1cU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v113;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1cU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v114;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1cU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v115;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1dU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v116;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1dU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v117;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1dU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v118;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1dU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v119;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1eU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v120;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1eU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v121;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1eU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v122;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1eU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v123;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1fU][0U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v124;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1fU][1U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v125;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1fU][2U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v126;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table[0x1fU][3U] 
        = __VdlyVal__pucb_intf__DOT__field_table_i__DOT__r_field_table__v127;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U] 
        = (IData)((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                    [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                    [1U])) << 0x20U) 
                   | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [0U]))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U] 
        = (IData)(((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [1U])) << 0x20U) 
                    | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                      [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                      [0U]))) >> 0x20U));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
        = (IData)((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                    [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                    [3U])) << 0x20U) 
                   | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [2U]))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
        = (IData)(((((QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                     [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                     [3U])) << 0x20U) 
                    | (QData)((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__r_field_table
                                      [vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr]
                                      [2U]))) >> 0x20U));
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[0U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1671, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[0U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[0U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U];
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[1U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1735, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[1U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[1U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U];
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[2U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1799, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[2U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[2U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U];
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[3U])) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1863, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[3U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd[3U] 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U];
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag 
        = (((0xf000U & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
                        >> 0xfU)) | (0xf00U & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
                                               >> 0x13U))) 
           | ((0xf0U & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U] 
                        >> 0x17U)) | (0xfU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U] 
                                              >> 0x1bU))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
        = (IData)((((QData)((IData)((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U]))) 
                    << 0x1bU) | (QData)((IData)((0x7ffffffU 
                                                 & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U])))));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
        = ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
            << 0x16U) | (IData)(((((QData)((IData)(
                                                   (0x7ffffffU 
                                                    & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U]))) 
                                   << 0x1bU) | (QData)((IData)(
                                                               (0x7ffffffU 
                                                                & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U])))) 
                                 >> 0x20U)));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
        = ((0xfffe0000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]) 
           | (0x1ffffU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U] 
                          >> 0xaU)));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
        = ((0x1ffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]) 
           | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
              << 0x11U));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
        = (0xfffU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U] 
                     >> 0xfU));
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1609, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1617, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1625, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1633, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag;
    if ((0x7ffffffU & (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                       ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1393, 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U], 
                               vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
            = ((0xf8000000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]) 
               | (0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U]));
    }
    if ((0x7ffffffU & (((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                         << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                   >> 0x1bU)) ^ ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                                  << 5U) 
                                                 | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
                                                    >> 0x1bU))))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1447, 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                 << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                           >> 0x1bU)), 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                 << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
                                           >> 0x1bU)));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U] 
            = ((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[0U]) 
               | (0xf8000000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U]));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
            = ((0xffc00000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U]) 
               | (0x3fffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U]));
    }
    if ((0x7ffffffU & (((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                         << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                     >> 0x16U)) ^ (
                                                   (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                                    << 0xaU) 
                                                   | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                                      >> 0x16U))))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1501, 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                 << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                             >> 0x16U)), 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                 << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
                                             >> 0x16U)));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U] 
            = ((0x3fffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[1U]) 
               | (0xffc00000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U]));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
            = ((0xfffe0000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U]) 
               | (0x1ffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]));
    }
    if ((0x7ffffffU & (((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                         << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                     >> 0x11U)) ^ (
                                                   (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[3U] 
                                                    << 0xfU) 
                                                   | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                                      >> 0x11U))))) {
        VL_COV_TOGGLE_CHG_ST_I(27, vlSymsp->__Vcoverage + 1555, 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                                 << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                             >> 0x11U)), 
                               ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[3U] 
                                 << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
                                             >> 0x11U)));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U] 
            = ((0x1ffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[2U]) 
               | (0xfffe0000U & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U]));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag[3U] 
            = (0xfffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U]);
    }
    VL_ASSIGNBIT_II(0U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U]) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1649]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1650]);
                }(), 0U)));
    VL_ASSIGNBIT_II(1U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                      << 5U) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                                >> 0x1bU))) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1651]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1652]);
                }(), 0U)));
    VL_ASSIGNBIT_II(2U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                      << 0xaU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                                  >> 0x16U))) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1653]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1654]);
                }(), 0U)));
    VL_ASSIGNBIT_II(3U, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, 
                    (((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                                      << 0xfU) | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                                  >> 0x11U))) 
                      == vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag)
                      ? ([&]() {
                    ++(vlSymsp->__Vcoverage[1655]);
                }(), 1U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[1656]);
                }(), 0U)));
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1357, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1365, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1373, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1381, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__w_comp_ptag = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1641, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit;
    }
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 752, vlSelfRef.pucb_intf__DOT__w_comp_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 760, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 768, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 776, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag 
        = vlSelfRef.pucb_intf__DOT__w_comp_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1341, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit;
    }
    vlSelfRef.pucb_intf__DOT__w_comp_hit = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 980, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 988, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 996, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1004, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag)));
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_comp_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 744, vlSelfRef.pucb_intf__DOT__w_comp_hit, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_hit);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_hit 
            = vlSelfRef.pucb_intf__DOT__w_comp_hit;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit 
        = vlSelfRef.pucb_intf__DOT__w_comp_hit;
    if ((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                 ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1080, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfff0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if ((0xf0U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                  ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1088, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                >> 4U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 4U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xff0fU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf0U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if ((0xf00U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                   ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1096, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                >> 8U), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag) 
                                         >> 8U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xf0ffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf00U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if ((0xf000U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                    ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1104, 
                               ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                >> 0xcU), ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag) 
                                           >> 0xcU));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag 
            = ((0xfffU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag)) 
               | (0xf000U & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag)));
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1012, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1064, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__6(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__6\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__r_field_wen = vlSelfRef.pucb_intf__DOT__w_field_wen;
    vlSelfRef.pucb_intf__DOT__r_field_mshr = (1U & 
                                              ((IData)(vlSelfRef.pucb_intf__DOT__w_field_stall)
                                                ? ([&]() {
                    ++(vlSymsp->__Vcoverage[977]);
                }(), 0U) : ([&]() {
                    ++(vlSymsp->__Vcoverage[978]);
                }(), (IData)(vlSelfRef.pucb_intf__DOT__w_cb_miss))));
    if (vlSelfRef.pucb_intf__DOT__w_field_stall) {
        ++(vlSymsp->__Vcoverage[975]);
    }
    if ((1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__w_field_stall)))) {
        ++(vlSymsp->__Vcoverage[976]);
    }
    ++(vlSymsp->__Vcoverage[979]);
    vlSelfRef.pucb_intf__DOT__r_field_addr = vlSelfRef.pucb_intf__DOT__w_field_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__r_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 972, vlSelfRef.pucb_intf__DOT__r_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_wen 
            = vlSelfRef.pucb_intf__DOT__r_field_wen;
    }
    vlSelfRef.pucb_intf__DOT__o_field_wen = vlSelfRef.pucb_intf__DOT__r_field_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__r_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 906, vlSelfRef.pucb_intf__DOT__r_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_mshr 
            = vlSelfRef.pucb_intf__DOT__r_field_mshr;
    }
    vlSelfRef.pucb_intf__DOT__o_field_mshr = vlSelfRef.pucb_intf__DOT__r_field_mshr;
    if ((vlSelfRef.pucb_intf__DOT__r_field_addr ^ vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_addr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 908, vlSelfRef.pucb_intf__DOT__r_field_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__r_field_addr 
            = vlSelfRef.pucb_intf__DOT__r_field_addr;
    }
    vlSelfRef.pucb_intf__DOT__o_field_addr = vlSelfRef.pucb_intf__DOT__r_field_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_field_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 158, vlSelfRef.pucb_intf__DOT__o_field_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_wen 
            = vlSelfRef.pucb_intf__DOT__o_field_wen;
    }
    vlSelfRef.o_field_wen = vlSelfRef.pucb_intf__DOT__o_field_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_field_mshr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_mshr))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 92, vlSelfRef.pucb_intf__DOT__o_field_mshr, vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_mshr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_mshr 
            = vlSelfRef.pucb_intf__DOT__o_field_mshr;
    }
    vlSelfRef.o_field_mshr = vlSelfRef.pucb_intf__DOT__o_field_mshr;
    if ((vlSelfRef.pucb_intf__DOT__o_field_addr ^ vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_addr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 94, vlSelfRef.pucb_intf__DOT__o_field_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_field_addr 
            = vlSelfRef.pucb_intf__DOT__o_field_addr;
    }
    vlSelfRef.o_field_addr = vlSelfRef.pucb_intf__DOT__o_field_addr;
}

VL_INLINE_OPT void Vtop___024root___nba_sequent__TOP__7(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_sequent__TOP__7\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data 
        = (1U & (~ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume)));
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en 
        = (0xffffU & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag) 
                      << (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr)));
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data 
        = (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict) 
            << 7U) | (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set 
        = (0x1fU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr) 
                    >> 2U));
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr 
        = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data;
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en) 
                   << (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2092, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en))) {
        VL_COV_TOGGLE_CHG_ST_I(16, vlSymsp->__Vcoverage + 3202, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data))) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3184, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set))) {
        VL_COV_TOGGLE_CHG_ST_I(5, vlSymsp->__Vcoverage + 1657, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set;
    }
    if ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr 
         ^ vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 1927, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1991, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_comb__TOP__0(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_comb__TOP__0\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<4>/*127:0*/ __Vtemp_1;
    VlWide<4>/*127:0*/ __Vtemp_2;
    VlWide<4>/*127:0*/ __Vtemp_3;
    VlWide<4>/*127:0*/ __Vtemp_4;
    // Body
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table
        [vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr];
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data))) {
        VL_COV_TOGGLE_CHG_ST_I(8, vlSymsp->__Vcoverage + 3152, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr 
        = (0x7fU & (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data));
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null 
        = (1U & ((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data) 
                 >> 7U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3170, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3168, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null 
        = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3122, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3120, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null, vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null = vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 838, vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 852, vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null 
        = vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 384, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_rev_ptr = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3946, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag 
        = ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[4178]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[4179]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 398, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_rev_ptr_null = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3944, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null 
        = ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume)
            ? ([&]() {
                ++(vlSymsp->__Vcoverage[4176]);
            }(), (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null))
            : ([&]() {
                ++(vlSymsp->__Vcoverage[4177]);
            }(), 0U));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 4152, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4150, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 4040, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4038, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null;
    }
    vlSelfRef.pucb_intf__DOT__wo_validity_table_null 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 728, vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wo_validity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 742, vlSelfRef.pucb_intf__DOT__wo_validity_table_null, vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__wo_validity_table_null 
            = vlSelfRef.pucb_intf__DOT__wo_validity_table_null;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_null;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null 
        = vlSelfRef.pucb_intf__DOT__wo_validity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 346, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag;
    }
    vlSelfRef.dbg_pucb_intf_wo_validity_table_vtag 
        = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 2032, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 360, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null;
    }
    vlSelfRef.dbg_pucb_intf_wo_validity_table_null 
        = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2046, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null 
        = ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall) 
           | (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2350, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null;
    }
    __Vtemp_1[0U] = 0U;
    __Vtemp_1[1U] = 0U;
    __Vtemp_1[2U] = 0U;
    __Vtemp_1[3U] = 0U;
    __Vtemp_2[0U] = 1U;
    __Vtemp_2[1U] = 0U;
    __Vtemp_2[2U] = 0U;
    __Vtemp_2[3U] = 0U;
    VL_SHIFTL_WWI(128,128,7, __Vtemp_3, __Vtemp_2, (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr));
    VL_COND_WIWW(128, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en, (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null), 
                 ([&]() {
                ++(vlSymsp->__Vcoverage[2609]);
            }(), __Vtemp_1), ([&]() {
                ++(vlSymsp->__Vcoverage[2608]);
            }(), __Vtemp_3));
    __Vtemp_4[0U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[0U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[0U]);
    __Vtemp_4[1U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[1U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[1U]);
    __Vtemp_4[2U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[2U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[2U]);
    __Vtemp_4[3U] = (vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[3U] 
                     ^ vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[3U]);
    if (__Vtemp_4) {
        VL_COV_TOGGLE_CHG_ST_W(128, vlSymsp->__Vcoverage + 2094, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[0U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[0U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[1U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[1U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[2U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[2U];
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en[3U] 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en[3U];
    }
}

VL_INLINE_OPT void Vtop___024root___nba_comb__TOP__1(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_comb__TOP__1\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag 
        = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                   >> (0xfU & VL_SHIFTL_III(4,4,32, (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru), 2U))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1349, vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag, vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag);
        vlSelfRef.pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag = vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 784, vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_lru_ptag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_comp_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1032, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1056, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag;
    }
}

VL_INLINE_OPT void Vtop___024root___nba_comb__TOP__2(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_comb__TOP__2\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__w_cb_miss = (1U & (~ 
                                                 (0U 
                                                  != 
                                                  ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_hit) 
                                                   & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_pvld)))));
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 820, vlSelfRef.pucb_intf__DOT__w_cb_miss, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_miss);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_miss 
            = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss 
        = vlSelfRef.pucb_intf__DOT__w_cb_miss;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 382, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_miss = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss;
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2654, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld 
        = ((~ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall)) 
           & (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3830, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen 
        = ((~ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss)) 
           & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen));
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2020, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2752, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld;
    }
    vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen 
        = ((IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld) 
           << (IData)(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr));
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4166, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen;
    if ((vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen 
         ^ vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen)) {
        VL_COV_TOGGLE_CHG_ST_I(32, vlSymsp->__Vcoverage + 2688, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen, vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen);
        vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen 
            = vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4054, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen;
    }
    vlSelfRef.pucb_intf__DOT__w_mem_wen = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 896, vlSelfRef.pucb_intf__DOT__w_mem_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_wen 
            = vlSelfRef.pucb_intf__DOT__w_mem_wen;
    }
    vlSelfRef.pucb_intf__DOT__o_mem_wen = vlSelfRef.pucb_intf__DOT__w_mem_wen;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_mem_wen) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_wen))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 174, vlSelfRef.pucb_intf__DOT__o_mem_wen, vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_wen);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_wen 
            = vlSelfRef.pucb_intf__DOT__o_mem_wen;
    }
    vlSelfRef.o_mem_wen = vlSelfRef.pucb_intf__DOT__o_mem_wen;
}

VL_INLINE_OPT void Vtop___024root___nba_comb__TOP__3(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_comb__TOP__3\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag;
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru;
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 0U;
    if ((1U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 0U;
        ++(vlSymsp->__Vcoverage[1124]);
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 1U;
    if ((2U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        ++(vlSymsp->__Vcoverage[1124]);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                       >> 4U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 1U;
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 2U;
    if ((4U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        ++(vlSymsp->__Vcoverage[1124]);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                       >> 8U));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 2U;
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 3U;
    if ((8U & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
               & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)))) {
        ++(vlSymsp->__Vcoverage[1124]);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag 
            = (0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                       >> 0xcU));
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line = 3U;
    } else {
        ++(vlSymsp->__Vcoverage[1125]);
    }
    ++(vlSymsp->__Vcoverage[1129]);
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k = 4U;
    if ((1U & (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
                & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit)) 
               >> (3U & vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k)))) {
        ++(vlSymsp->__Vcoverage[1126]);
    }
    if ((1U & (~ ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit) 
                  >> (3U & vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k))))) {
        ++(vlSymsp->__Vcoverage[1127]);
    }
    if ((1U & (~ ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld) 
                  >> (3U & vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k))))) {
        ++(vlSymsp->__Vcoverage[1128]);
    }
    ++(vlSymsp->__Vcoverage[1130]);
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume 
        = ((~ ((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data) 
               >> (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru))) 
           & ((~ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall)) 
              & (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss)));
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1116, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1112, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line;
    }
    vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line 
        = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2068, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume 
        = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 1044, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_ctag = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 1040, vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line, vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line);
        vlSelfRef.pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line 
            = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_cline = vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line;
    if (((IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 2050, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume, vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume);
        vlSelfRef.pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume 
            = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_consume = vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 862, vlSelfRef.pucb_intf__DOT__w_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag 
        = vlSelfRef.pucb_intf__DOT__w_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_cline) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_cline))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 858, vlSelfRef.pucb_intf__DOT__w_cb_cline, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_cline);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_cline 
            = vlSelfRef.pucb_intf__DOT__w_cb_cline;
    }
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline 
        = vlSelfRef.pucb_intf__DOT__w_cb_cline;
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline 
        = vlSelfRef.pucb_intf__DOT__w_cb_cline;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 854, vlSelfRef.pucb_intf__DOT__w_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_cb_consume 
            = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume 
        = vlSelfRef.pucb_intf__DOT__w_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3582, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 408, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_ctag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 3900, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 404, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_cline = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 3908, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag 
        = (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set) 
            << 2U) | (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline));
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 3580, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 400, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume;
    }
    vlSelfRef.dbg_pucb_intf_w_cb_consume = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume;
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume))) {
        VL_COV_TOGGLE_CHG_ST_I(1, vlSymsp->__Vcoverage + 4184, vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 4168, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 4064, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 4056, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr;
    }
    vlSelfRef.pucb_intf__DOT__w_mem_addr = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3960, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag, vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__w_field_table_vtag = vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 898, vlSelfRef.pucb_intf__DOT__w_mem_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_mem_addr 
            = vlSelfRef.pucb_intf__DOT__w_mem_addr;
    }
    vlSelfRef.pucb_intf__DOT__o_mem_addr = vlSelfRef.pucb_intf__DOT__w_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__w_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 572, vlSelfRef.pucb_intf__DOT__w_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__w_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__w_field_table_vtag;
    }
    vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__w_field_table_vtag;
    vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag 
        = vlSelfRef.pucb_intf__DOT__w_field_table_vtag;
    if (((IData)(vlSelfRef.pucb_intf__DOT__o_mem_addr) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_addr))) {
        VL_COV_TOGGLE_CHG_ST_I(4, vlSymsp->__Vcoverage + 176, vlSelfRef.pucb_intf__DOT__o_mem_addr, vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_addr);
        vlSelfRef.pucb_intf__DOT____Vtogcov__o_mem_addr 
            = vlSelfRef.pucb_intf__DOT__o_mem_addr;
    }
    vlSelfRef.o_mem_addr = vlSelfRef.pucb_intf__DOT__o_mem_addr;
    if (((IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 3502, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag, vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag);
        vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag;
    }
    if (((IData)(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag))) {
        VL_COV_TOGGLE_CHG_ST_I(7, vlSymsp->__Vcoverage + 254, vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag, vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag);
        vlSelfRef.pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag 
            = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag;
    }
    vlSelfRef.dbg_pucb_intf_w_field_table_vtag = vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag;
}
