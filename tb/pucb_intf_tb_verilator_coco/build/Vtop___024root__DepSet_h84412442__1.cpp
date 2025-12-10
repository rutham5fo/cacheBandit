// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

VL_INLINE_OPT void Vtop___024root___nba_comb__TOP__4(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___nba_comb__TOP__4\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
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
    if (((IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset) 
         ^ (IData)(vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset))) {
        VL_COV_TOGGLE_CHG_ST_I(2, vlSymsp->__Vcoverage + 4188, vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset, vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset);
        vlSelfRef.pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset 
            = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset;
    }
    vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset = vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset;
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
}
