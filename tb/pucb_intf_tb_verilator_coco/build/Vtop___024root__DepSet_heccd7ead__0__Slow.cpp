// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop___024root.h"

VL_ATTR_COLD void Vtop___024root___eval_static(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_static\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
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
}

VL_ATTR_COLD void Vtop___024root___eval_initial__TOP(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_initial(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_initial\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtop___024root___eval_initial__TOP(vlSelf);
}

VL_ATTR_COLD void Vtop___024root___eval_final(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_final\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(Vtop___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtop___024root___eval_phase__stl(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_settle(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_settle\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY(((0x64U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtop___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("/mnt/c/Users/91988/Documents/Amruth/Files/Projects/CacheBandit/tb/pucb_intf_tb_verilator_coco/debug/hdl/pucb_intf.sv", 35, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtop___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelfRef.__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__stl(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__stl\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VstlTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

void Vtop___024root___ico_sequent__TOP__0(Vtop___024root* vlSelf);

VL_ATTR_COLD void Vtop___024root___eval_stl(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_stl\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        Vtop___024root___ico_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void Vtop___024root___eval_triggers__stl(Vtop___024root* vlSelf);

VL_ATTR_COLD bool Vtop___024root___eval_phase__stl(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___eval_phase__stl\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtop___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelfRef.__VstlTriggered.any();
    if (__VstlExecute) {
        Vtop___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__ico(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__ico\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VicoTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VicoTriggered.word(0U))) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__act(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__act\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge pucb_intf.i_pu_clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(posedge pucb_intf.field_table_i.i_clk)\n");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @(posedge pucb_intf.validity_table_i.i_clk)\n");
    }
    if ((8ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 3 is active: @(posedge pucb_intf.lru_regfile_i.i_clk)\n");
    }
    if ((0x10ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 4 is active: @(posedge pucb_intf.rev_ptr_table_i.i_clk)\n");
    }
    if ((0x20ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 5 is active: @(posedge pucb_intf.in_cdc_i.i_clk)\n");
    }
    if ((0x40ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 6 is active: @(posedge pucb_intf.controller_i.i_clk)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop___024root___dump_triggers__nba(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___dump_triggers__nba\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge pucb_intf.i_pu_clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(posedge pucb_intf.field_table_i.i_clk)\n");
    }
    if ((4ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @(posedge pucb_intf.validity_table_i.i_clk)\n");
    }
    if ((8ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 3 is active: @(posedge pucb_intf.lru_regfile_i.i_clk)\n");
    }
    if ((0x10ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 4 is active: @(posedge pucb_intf.rev_ptr_table_i.i_clk)\n");
    }
    if ((0x20ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 5 is active: @(posedge pucb_intf.in_cdc_i.i_clk)\n");
    }
    if ((0x40ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 6 is active: @(posedge pucb_intf.controller_i.i_clk)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root___ctor_var_reset\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->i_pu_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1038980382499815780ull);
    vlSelf->i_cb_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 179726239762199668ull);
    vlSelf->i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9693334148897220726ull);
    vlSelf->i_cb_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14379834473643716603ull);
    vlSelf->i_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9590303701480894155ull);
    vlSelf->i_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8315480492740760090ull);
    vlSelf->i_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15956655324120854664ull);
    vlSelf->i_cb_consume_buf = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 2089789531748228213ull);
    vlSelf->o_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8038812825025512201ull);
    vlSelf->o_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10622246375800844294ull);
    vlSelf->o_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13231281692083362432ull);
    vlSelf->o_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6236046167910508582ull);
    vlSelf->o_cb_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 9102279328877716819ull);
    vlSelf->o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15541171249666660861ull);
    vlSelf->o_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15128744791701412155ull);
    vlSelf->o_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13252764739101281956ull);
    vlSelf->dbg_pucb_intf_w_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9714299573724362175ull);
    vlSelf->dbg_pucb_intf_w_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17034217879341642207ull);
    vlSelf->dbg_pucb_intf_w_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2188301996436246745ull);
    vlSelf->dbg_pucb_intf_w_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 11547025195993075792ull);
    vlSelf->dbg_pucb_intf_w_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 157286681140710347ull);
    vlSelf->dbg_pucb_intf_w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 10159714822288613393ull);
    vlSelf->dbg_pucb_intf_w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 181943591612107164ull);
    vlSelf->dbg_pucb_intf_wo_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 4386022105661477194ull);
    vlSelf->dbg_pucb_intf_wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 952658278550637225ull);
    vlSelf->dbg_pucb_intf_wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7816848204919845003ull);
    vlSelf->dbg_pucb_intf_w_cb_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1583933695251501304ull);
    vlSelf->dbg_pucb_intf_w_cb_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 2514486812929808844ull);
    vlSelf->dbg_pucb_intf_w_cb_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9998964985965058163ull);
    vlSelf->dbg_pucb_intf_w_cb_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16331310027524387470ull);
    vlSelf->dbg_pucb_intf_w_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1045875187203843951ull);
    vlSelf->dbg_pucb_intf_w_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 13647476582158930312ull);
    vlSelf->dbg_pucb_intf_w_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9981102115827253034ull);
    vlSelf->dbg_pucb_intf_w_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5193509918911176736ull);
    vlSelf->dbg_pucb_intf_wo_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15717162122993977473ull);
    vlSelf->dbg_pucb_intf_w_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 10149592978006360397ull);
    vlSelf->dbg_pucb_intf_w_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 12402582307771057191ull);
    vlSelf->dbg_pucb_intf_wo_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 15077753762532131905ull);
    vlSelf->dbg_pucb_intf_w_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7101438300035405654ull);
    vlSelf->dbg_pucb_intf_w_cb_consume_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6405813815520797347ull);
    vlSelf->dbg_pucb_intf_wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4737521748986792374ull);
    vlSelf->pucb_intf__DOT__i_pu_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4516981900952463179ull);
    vlSelf->pucb_intf__DOT__i_cb_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2100468178218711827ull);
    vlSelf->pucb_intf__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2047377127274571139ull);
    vlSelf->pucb_intf__DOT__i_cb_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12524375299971092307ull);
    vlSelf->pucb_intf__DOT__i_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3971831283190612106ull);
    vlSelf->pucb_intf__DOT__i_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8963354107340179699ull);
    vlSelf->pucb_intf__DOT__i_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12761248908886504323ull);
    vlSelf->pucb_intf__DOT__i_cb_consume_buf = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6714790114197364405ull);
    vlSelf->pucb_intf__DOT__o_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5504393808854182179ull);
    vlSelf->pucb_intf__DOT__o_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16139024056415443425ull);
    vlSelf->pucb_intf__DOT__o_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12905764921461551632ull);
    vlSelf->pucb_intf__DOT__o_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14168472378758056815ull);
    vlSelf->pucb_intf__DOT__o_cb_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13679435394112848551ull);
    vlSelf->pucb_intf__DOT__o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9882232218001396188ull);
    vlSelf->pucb_intf__DOT__o_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17613446739066590675ull);
    vlSelf->pucb_intf__DOT__o_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4143364650129665937ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5228599257923956205ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11745721672254243586ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10895480842364674501ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 9865706007892213428ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 5624232516508018193ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 8925756304377715245ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4530809574193977703ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 15914054494375409694ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 3060045115117679375ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9055530780332092046ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17662587049996194170ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 15491556483269247451ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11065094144764400383ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 17546013369295328126ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10411215412849972920ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 7419399517412584218ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4607169048407915246ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10385744151798621882ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11410069390606483950ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17606412011823073223ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8695488602080762700ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5809203799492577688ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7514834764719065280ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12965157549969152743ull);
    vlSelf->pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5531406143356622398ull);
    vlSelf->pucb_intf__DOT__w_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12780018209801243530ull);
    vlSelf->pucb_intf__DOT__w_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7189611068874734260ull);
    vlSelf->pucb_intf__DOT__w_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17241609258043249268ull);
    vlSelf->pucb_intf__DOT__w_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5764734770545409515ull);
    vlSelf->pucb_intf__DOT__w_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 1516211111581190230ull);
    vlSelf->pucb_intf__DOT__w_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 9272833211819130572ull);
    vlSelf->pucb_intf__DOT__w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 146705553116435129ull);
    vlSelf->pucb_intf__DOT__w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3441937026069515908ull);
    vlSelf->pucb_intf__DOT__wo_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 10009194329500621278ull);
    vlSelf->pucb_intf__DOT__wo_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9195145964324361991ull);
    vlSelf->pucb_intf__DOT__wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 3476486464345831111ull);
    vlSelf->pucb_intf__DOT__wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17016743096385897852ull);
    vlSelf->pucb_intf__DOT__w_comp_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4548909877296603701ull);
    vlSelf->pucb_intf__DOT__w_comp_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 365641007553449914ull);
    vlSelf->pucb_intf__DOT__w_comp_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2906376858529326758ull);
    vlSelf->pucb_intf__DOT__w_comp_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11866521516873863343ull);
    vlSelf->pucb_intf__DOT__w_cb_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4486305921868519588ull);
    vlSelf->pucb_intf__DOT__w_cb_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 14270599958754652769ull);
    vlSelf->pucb_intf__DOT__w_cb_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1486621400315745358ull);
    vlSelf->pucb_intf__DOT__w_cb_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 15072225391509760173ull);
    vlSelf->pucb_intf__DOT__w_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3868150483120188145ull);
    vlSelf->pucb_intf__DOT__w_cb_consume_buf = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11890299876602720081ull);
    vlSelf->pucb_intf__DOT__w_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 312011693844899226ull);
    vlSelf->pucb_intf__DOT__w_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6238333479627536300ull);
    vlSelf->pucb_intf__DOT__w_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16586453609798462031ull);
    vlSelf->pucb_intf__DOT__wo_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6710632334668109839ull);
    vlSelf->pucb_intf__DOT__w_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11285199680968687958ull);
    vlSelf->pucb_intf__DOT__w_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4883683867455278643ull);
    vlSelf->pucb_intf__DOT__wo_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11075852562842859116ull);
    vlSelf->pucb_intf__DOT__w_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6715226698475942326ull);
    vlSelf->pucb_intf__DOT__wo_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 661414022950547079ull);
    vlSelf->pucb_intf__DOT__w_cb_consume_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12163501122543832189ull);
    vlSelf->pucb_intf__DOT__wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17720125549564989916ull);
    vlSelf->pucb_intf__DOT__w_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17439106759398673715ull);
    vlSelf->pucb_intf__DOT__w_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5415342351839822348ull);
    vlSelf->pucb_intf__DOT__r_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17036565816451982247ull);
    vlSelf->pucb_intf__DOT__r_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5025210076765765422ull);
    vlSelf->pucb_intf__DOT__r_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10912027980031436935ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_pu_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6917581520092561269ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_cb_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11049779222342389190ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6578576852043778173ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_cb_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4616794813802564996ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6895208114105042529ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17860434994974764396ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4276688701317250648ull);
    vlSelf->pucb_intf__DOT____Vtogcov__i_cb_consume_buf = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 1627017185538102052ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10711161797058951616ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15986933548855399140ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11989832072067302200ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11536584609564839191ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_cb_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17822456555153953665ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3597696444690616715ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17555741200158374402ull);
    vlSelf->pucb_intf__DOT____Vtogcov__o_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 18435588989689716894ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11422233042553593943ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6068312721997408319ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18106157354838832699ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 12375647895544882565ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 9912272340328258521ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 2015514788686541824ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12821782810134106253ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 753155606693568721ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 11822666712045349588ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13504293881997375582ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12048292636348736538ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 8602115694931989201ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3638929178032284811ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16905005287294619499ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13757703348323375705ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 10038127008763155922ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1631024966707415834ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10760847625965977376ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12320210858988310297ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17274438889065416978ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 10065052695758799118ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 1083407940317412803ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 5722866007790499314ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_w_cb_consume_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11311728515179486859ull);
    vlSelf->pucb_intf__DOT____Vtogcov__dbg_pucb_intf_wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17976251522746971335ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9916141886463621529ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17782045647208989482ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4755360807818433285ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6798118691367042168ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 15876747873583490869ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 13803380176544301979ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 7299508136008776858ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13918293190222778949ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 16854412359056893649ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6100176416880548737ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 9716940983880236924ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16287818219157043091ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_comp_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 7546600178070322761ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_comp_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15972966496445069277ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_comp_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4532209865181721234ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_comp_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 15223944690961889228ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 2459737854282312874ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 3290506273395471324ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 5649589800634336858ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 18307435351491314582ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14638826938565395984ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_consume_buf = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6413858845232700530ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 13651114345969236594ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 74387800898649073ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16553425602655384309ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8122913665813145414ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 7690395788043667877ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6833017916258012028ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11040439199539152926ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11907401881711800373ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11425184530017533181ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_cb_consume_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5678480617383187359ull);
    vlSelf->pucb_intf__DOT____Vtogcov__wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 16069290238610610644ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13960051225140105571ull);
    vlSelf->pucb_intf__DOT____Vtogcov__w_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6528485411433823300ull);
    vlSelf->pucb_intf__DOT____Vtogcov__r_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5652665365075722248ull);
    vlSelf->pucb_intf__DOT____Vtogcov__r_field_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7667336833529607825ull);
    vlSelf->pucb_intf__DOT____Vtogcov__r_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 120130049912722703ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__i_field_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 12557072146487949413ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__i_field_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11001111059270653308ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__i_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17884266658408521851ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__i_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1342883016853360437ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13507364746249448415ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__o_cb_line = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 18098922358718387847ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__o_cb_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17281907353981354641ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15695422295389431167ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 3883677122708016389ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_field_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 152104501588481086ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13299999029457556239ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_field_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10555014080842915640ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_comp_line = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12286147819889414827ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__w_comp_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 9421200699374229253ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT__k = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16182328150064705658ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9257039051862152094ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13031722405666579450ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17830684780019321051ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13061867774859680980ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__i_field_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 18375148829405096122ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_line = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 10092323670721937978ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__o_cb_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17463729762562804815ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 8804278345682281683ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2908952910885988643ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 14301083247363521125ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8659013430800041665ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_field_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13326717773334344704ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_line = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9465846566308188081ull);
    vlSelf->pucb_intf__DOT__comparator_i__DOT____Vtogcov__w_comp_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 9281811312967277332ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 12377996818981492481ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 12521347699846992735ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 10486818885112034718ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__read_ptr__Vstatic__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 548363158234425804ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__read_ptr__Vstatic__k = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11978472844973602639ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__w_nactv = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 14742335846117224221ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__w_tbits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 7593567098817550955ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__w_pactv = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 10461272992713686933ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16013408733737626269ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 11765327652288774554ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 10590391427773660658ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__i_field_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 6184884416361162791ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 15945308791922760732ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__o_field_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17499065185105009725ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_nactv = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16629869489898566393ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_tbits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 5574338547151886728ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_pactv = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 14404747409153166002ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 16116183009751154763ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 6936735903587108786ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__w_field_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4925900909906968669ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__0__KET__w_tread = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8760775491631155654ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT____Vtogcov__gen_lru_nxt__BRA__1__KET__w_tread = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 8549223586266661568ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 250351012562691688ull);
    vlSelf->pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1860511013884278648ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5377060246458641402ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5631655714044795437ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 16005401957564699784ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 17316720278071555054ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_rd_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 4297560847676764168ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_wr_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 2691315094507341880ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7469959010657417058ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__i_wr_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7006242161871753925ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__o_comp_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 10241940851913933925ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 16680884192106903255ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__o_comp_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 13855632915133937491ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1258931433635394341ull);
    VL_SCOPED_RAND_RESET_W(108, vlSelf->pucb_intf__DOT__field_table_i__DOT__w_field_atag, __VscopeHash, 11136183116806136070ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_field_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6469275983694318628ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_tag_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8677416043196550048ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__k = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1170128676590237768ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__m = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10346591630788210720ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_wr_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 8304032013878806498ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_wr_line = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6098705850135723912ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->pucb_intf__DOT__field_table_i__DOT__w_set_rd, __VscopeHash, 5529089993634709410ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_line_wr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5567787536587296239ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT__w_line_wen = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5261791274356260478ull);
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        for (int __Vi1 = 0; __Vi1 < 4; ++__Vi1) {
            vlSelf->pucb_intf__DOT__field_table_i__DOT__r_field_table[__Vi0][__Vi1] = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 2845807997133132920ull);
        }
    }
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11649562193058025224ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3146487126002604185ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 16126606902295911414ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15268949945410988240ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_rd_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 2885732482580129898ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 11112704373492261625ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6643480439206849239ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__i_wr_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15622477314509249680ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 6226932623319475845ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 15450204625153427697ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__o_comp_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2648394951614196697ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13405440649618734416ull);
    VL_SCOPED_RAND_RESET_W(108, vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_atag, __VscopeHash, 1919350199091553608ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_field_ptag = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 5413783995142313828ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_tag_hit = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4934348283951256202ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 12560602503714171190ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_wr_line = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 16290946396218947475ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_set_rd, __VscopeHash, 15322218037455510874ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10934663949523719753ull);
    vlSelf->pucb_intf__DOT__field_table_i__DOT____Vtogcov__w_line_wen = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 1436424653927749468ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1849760517510602383ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13791315002101696380ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8932682037008502898ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11191650435015302252ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4948882041043585415ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_field_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1031188739167253036ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_rd_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 2706203822662305791ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_wr_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 15419181882770351622ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_wr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7104651041091564134ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__i_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14745488872297976571ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__o_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9570766617394726348ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__o_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8239546050121658771ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 7708734465169543564ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__wo_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13448463016797707317ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__w_rd_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 12390921643215416838ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__w_rd_data = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 17102252243004313944ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__w_wr_data = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8613523764684430658ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->pucb_intf__DOT__validity_table_i__DOT__w_wr_en, __VscopeHash, 4729875651454933367ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__w_wr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9895017333402318194ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->pucb_intf__DOT__validity_table_i__DOT__r_vld_table, __VscopeHash, 7830590802682741865ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT__k = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1437631128702758849ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14477533301484654785ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2613601996076421888ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7091653027373596355ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_mshr = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16152496906253247467ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_lru = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13917905068897401647ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5120870113616923831ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_rd_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 8872502497756118553ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 7698216335218230034ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_wr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11819612150856466526ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__i_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12831860582607043236ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9972774583011638352ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__o_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2641182149075603354ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_pvld = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13103461858088051055ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__wo_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6102171634358407306ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_addr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 13973430209062071417ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_rd_data = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5122927140970839386ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_data = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11880921882705671308ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_en, __VscopeHash, 2045041487389833572ull);
    vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__w_wr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7661578411400384066ull);
    VL_SCOPED_RAND_RESET_W(128, vlSelf->pucb_intf__DOT__validity_table_i__DOT____Vtogcov__r_vld_table, __VscopeHash, 7959697320398007901ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18123597274733801459ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12295915782671142349ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 14115718409636121186ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 3955488908465083138ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 5446005466936071433ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 10428726646185438396ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16851505962615729663ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 18291135213768571713ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 14718215290976133255ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3343652917796832816ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__k = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6402981858235584841ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 16420789588391032682ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 1877361345730839501ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3241969945149445358ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1273193745181192988ull);
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[__Vi0] = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 13537171894036294705ull);
    }
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7560142287121204988ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5845750105219405581ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_rd_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 2456129598665130813ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_wr_addr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 1181138905548466780ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 13473936346469083588ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_nxt = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6632357373658779459ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_lru_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3302656558868921306ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__i_field_stall = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14843492366229028940ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur_bits = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 5766599033251447104ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__o_field_lru_cur = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 928561864412095410ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_rd = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 2551398469169038996ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wr = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 7784122495341779098ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_lru_wen = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10994705514360782867ull);
    vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__w_wr_vld = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2148688150377504804ull);
    for (int __Vi0 = 0; __Vi0 < 32; ++__Vi0) {
        vlSelf->pucb_intf__DOT__lru_regfile_i__DOT____Vtogcov__r_lru_state[__Vi0] = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 16444180066266059025ull);
    }
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3697463582120127016ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6077715929114247061ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 14147194078482908558ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5945336464366654826ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 16971024103930013879ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2906182724367649436ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 87439463003061011ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17266994180765510525ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 4923564605171958892ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4121963378276632299ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5437468435334072548ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15039698502758378568ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3730903729792440822ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 17963564929811248076ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 14519700003112392721ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2352788169133401220ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4497136695433707879ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 7558677238862533206ull);
    }
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT__k = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5019070971534330170ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17157450621389548613ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16536179023322399814ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_rd_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11393310780221932145ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 12707691990854884749ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_wr_data = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 2358101558885992136ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_evict = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7610989596297867853ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__i_field_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2466527778032904324ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14253902341073936544ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__o_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 11744666599568194487ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2758429522962652902ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11448937239517718334ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rd_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 8044410701213809398ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5853679255384782913ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 9462076671887310783ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_data = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 9085994619721200829ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en_flag = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1987930031137734882ull);
    vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__w_wr_en = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 11435683051860393438ull);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->pucb_intf__DOT__rev_ptr_table_i__DOT____Vtogcov__r_ptr_table[__Vi0] = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 6308264540868511055ull);
    }
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3326044566589280910ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17452893471560210968ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 2841220772993217387ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1428895138317726514ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1550643308592502106ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 5610744461644232448ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11253837068875204855ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 160375404176120794ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4211266639168567705ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11337055791298044457ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 12226418824869078181ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 522875342663459054ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4654072978045731779ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__w_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3349547893922841209ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 1261909431516766464ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__r_cntr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 4883175521857982060ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 16202425830009045731ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6960482078304348764ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12501591987051715819ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 2696678069102789122ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11873253487417620699ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7033972965652004606ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1902606048124337162ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 13692750463309652068ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6281630601465747215ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16608981887187720586ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 1457035100317854130ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__i_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15313468636969938675ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 663966518689707787ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 18194102639202531072ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10492677850703383125ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 8181066846072453507ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 10286625062339783915ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 15400477146268535443ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__w_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8734804072433359539ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__wo_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 13195095848709407989ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cntr = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 10975978954463019479ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 1741870392528100022ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8242821927844216997ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1364384573622395147ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 10127678591034294482ull);
    vlSelf->pucb_intf__DOT__in_cdc_i__DOT____Vtogcov__r_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13307855827878118568ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 94444216053772679ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16941456673661091675ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10801602516443908284ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 14452780915605176179ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 5355727414468994499ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 10637154613142720106ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 9192751066485564862ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 708873801099862588ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5331356529262878561ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 8240547829698855189ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8944433027548378715ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 12324644072370667574ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 4482305787280564795ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14658526565429835746ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8185008417904780158ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 13384943440499931426ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8855097776302173513ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4902069000778039408ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 17173043035453491422ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15930184333505570749ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 7323354352393591990ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17859683794515891599ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 5757084674481308925ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9241214085058822409ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 9308474794611166449ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_miss = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7733303316036200376ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ocb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9506248705415739377ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11794255399712383948ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_atag = VL_SCOPED_RAND_RESET_I(27, __VscopeHash, 13238374310926235321ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_field_set = VL_SCOPED_RAND_RESET_I(5, __VscopeHash, 2067196741692004519ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_ctag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 9821499072871331656ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_cline = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1562492439842979412ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 13968075420621131156ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16116870805995259247ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_ofield_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 7215985083288803493ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17306645072657647391ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__i_cb_rev_ptr = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 4171889847689080857ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 9742142641088637788ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10993321917337353926ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 908425260888900591ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_ovalidity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 15373060270116730061ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7464201237666187481ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__o_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 4391827927382964050ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 2484457969420715703ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_field_table_wr_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15265811084985678832ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_cb_lru_ptag = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 15062687678809994880ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_null = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13163233192866929013ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__wo_validity_table_vtag = VL_SCOPED_RAND_RESET_I(7, __VscopeHash, 10588100015209778393ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_wen = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 507558071984369329ull);
    vlSelf->pucb_intf__DOT__wb_gen_i__DOT____Vtogcov__w_mem_addr = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 11260521938564889710ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6677076072816759638ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5456021828328798319ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__i_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13300674342382655865ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11566181028204892790ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 12767150705704644287ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10821270683020231510ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 13848108495337458475ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2710978818136936499ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__w_update_stg_tap = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7981010937046495894ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9286558925199615783ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 11761168131335424177ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2089697471606903614ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6522436975442811656ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__i_clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11195268868598440759ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__i_rst = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13402996249749709074ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__i_cb_consume = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7548309998175778409ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6323301583964144496ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__o_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6816120461376783654ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11586974146914564464ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__w_nxt_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 16889526992651915328ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__w_dec_stg_tap = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17200800470442021308ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__w_update_stg_tap = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10743020541995257890ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_dec_stg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2288068845863292748ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_consume_update_stg = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3438969867679536651ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_lru_sel = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2650448268778497518ull);
    vlSelf->pucb_intf__DOT__controller_i__DOT____Vtogcov__r_cb_vtp_offset = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 6029974988919055146ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__i_pu_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3445684967705381550ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__field_table_i__DOT__i_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7413824488795435963ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__validity_table_i__DOT__i_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1337856254635863457ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__lru_regfile_i__DOT__i_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15443517176468934272ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17642647895284876781ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__in_cdc_i__DOT__i_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9896457616368002526ull);
    vlSelf->__Vtrigprevexpr___TOP__pucb_intf__DOT__controller_i__DOT__i_clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8248354696933020936ull);
}
