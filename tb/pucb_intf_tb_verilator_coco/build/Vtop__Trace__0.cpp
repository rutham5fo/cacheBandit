// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_fst_c.h"
#include "Vtop__Syms.h"


void Vtop___024root__trace_chg_0_sub_0(Vtop___024root* vlSelf, VerilatedFst::Buffer* bufp);

void Vtop___024root__trace_chg_0(void* voidSelf, VerilatedFst::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_0\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtop___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtop___024root__trace_chg_0_sub_0(Vtop___024root* vlSelf, VerilatedFst::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_chg_0_sub_0\n"); );
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    bufp->chgBit(oldp+0,(vlSelfRef.i_pu_clk));
    bufp->chgBit(oldp+1,(vlSelfRef.i_cb_clk));
    bufp->chgBit(oldp+2,(vlSelfRef.i_rst));
    bufp->chgBit(oldp+3,(vlSelfRef.i_cb_en));
    bufp->chgBit(oldp+4,(vlSelfRef.i_field_mshr));
    bufp->chgBit(oldp+5,(vlSelfRef.i_field_wen));
    bufp->chgIData(oldp+6,(vlSelfRef.i_field_addr),32);
    bufp->chgCData(oldp+7,((0xfU & (IData)(vlSelfRef.i_cb_consume_buf))),4);
    bufp->chgCData(oldp+8,((0xfU & ((IData)(vlSelfRef.i_cb_consume_buf) 
                                    >> 4U))),4);
    bufp->chgBit(oldp+9,(vlSelfRef.o_field_mshr));
    bufp->chgIData(oldp+10,(vlSelfRef.o_field_addr),32);
    bufp->chgBit(oldp+11,(vlSelfRef.o_field_wen));
    bufp->chgBit(oldp+12,(vlSelfRef.o_cb_consume));
    bufp->chgCData(oldp+13,(vlSelfRef.o_cb_ptag),4);
    bufp->chgCData(oldp+14,(vlSelfRef.o_cb_vtp_offset),2);
    bufp->chgBit(oldp+15,(vlSelfRef.o_mem_wen));
    bufp->chgCData(oldp+16,(vlSelfRef.o_mem_addr),4);
    bufp->chgBit(oldp+17,(vlSelfRef.dbg_pucb_intf_w_field_stall));
    bufp->chgBit(oldp+18,(vlSelfRef.dbg_pucb_intf_w_field_wen));
    bufp->chgBit(oldp+19,(vlSelfRef.dbg_pucb_intf_w_field_mshr));
    bufp->chgIData(oldp+20,(vlSelfRef.dbg_pucb_intf_w_field_atag),27);
    bufp->chgCData(oldp+21,(vlSelfRef.dbg_pucb_intf_w_field_set),5);
    bufp->chgCData(oldp+22,(vlSelfRef.dbg_pucb_intf_w_field_table_vtag),7);
    bufp->chgIData(oldp+23,(vlSelfRef.dbg_pucb_intf_w_field_table_wr_data),32);
    bufp->chgCData(oldp+24,(vlSelfRef.dbg_pucb_intf_wo_field_table_vtag),7);
    bufp->chgCData(oldp+25,(vlSelfRef.dbg_pucb_intf_wo_validity_table_vtag),7);
    bufp->chgBit(oldp+26,(vlSelfRef.dbg_pucb_intf_wo_validity_table_null));
    bufp->chgCData(oldp+27,(vlSelfRef.dbg_pucb_intf_w_cb_lru_cur),2);
    bufp->chgCData(oldp+28,(vlSelfRef.dbg_pucb_intf_w_cb_lru_cur_bits),3);
    bufp->chgCData(oldp+29,(vlSelfRef.dbg_pucb_intf_w_cb_lru_nxt),2);
    bufp->chgCData(oldp+30,(vlSelfRef.dbg_pucb_intf_w_cb_lru_nxt_bits),3);
    bufp->chgBit(oldp+31,(vlSelfRef.dbg_pucb_intf_w_cb_miss));
    bufp->chgCData(oldp+32,(vlSelfRef.dbg_pucb_intf_w_cb_rev_ptr),7);
    bufp->chgBit(oldp+33,(vlSelfRef.dbg_pucb_intf_w_cb_rev_ptr_null));
    bufp->chgBit(oldp+34,(vlSelfRef.dbg_pucb_intf_w_cb_consume));
    bufp->chgBit(oldp+35,(vlSelfRef.dbg_pucb_intf_wo_cb_consume));
    bufp->chgCData(oldp+36,(vlSelfRef.dbg_pucb_intf_w_cb_cline),2);
    bufp->chgCData(oldp+37,(vlSelfRef.dbg_pucb_intf_w_cb_ctag),4);
    bufp->chgCData(oldp+38,(vlSelfRef.dbg_pucb_intf_wo_cb_ctag),4);
    bufp->chgCData(oldp+39,(vlSelfRef.dbg_pucb_intf_w_cb_vtp_offset),2);
    bufp->chgBit(oldp+40,(vlSelfRef.dbg_pucb_intf_w_cb_consume_sel));
    bufp->chgCData(oldp+41,(vlSelfRef.dbg_pucb_intf_wo_cb_lru_ptag),4);
    bufp->chgBit(oldp+42,(vlSelfRef.pucb_intf__DOT__i_pu_clk));
    bufp->chgBit(oldp+43,(vlSelfRef.pucb_intf__DOT__i_cb_clk));
    bufp->chgBit(oldp+44,(vlSelfRef.pucb_intf__DOT__i_rst));
    bufp->chgBit(oldp+45,(vlSelfRef.pucb_intf__DOT__i_cb_en));
    bufp->chgBit(oldp+46,(vlSelfRef.pucb_intf__DOT__i_field_mshr));
    bufp->chgBit(oldp+47,(vlSelfRef.pucb_intf__DOT__i_field_wen));
    bufp->chgIData(oldp+48,(vlSelfRef.pucb_intf__DOT__i_field_addr),32);
    bufp->chgCData(oldp+49,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf))),4);
    bufp->chgCData(oldp+50,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__i_cb_consume_buf) 
                                     >> 4U))),4);
    bufp->chgBit(oldp+51,(vlSelfRef.pucb_intf__DOT__o_field_mshr));
    bufp->chgIData(oldp+52,(vlSelfRef.pucb_intf__DOT__o_field_addr),32);
    bufp->chgBit(oldp+53,(vlSelfRef.pucb_intf__DOT__o_field_wen));
    bufp->chgBit(oldp+54,(vlSelfRef.pucb_intf__DOT__o_cb_consume));
    bufp->chgCData(oldp+55,(vlSelfRef.pucb_intf__DOT__o_cb_ptag),4);
    bufp->chgCData(oldp+56,(vlSelfRef.pucb_intf__DOT__o_cb_vtp_offset),2);
    bufp->chgBit(oldp+57,(vlSelfRef.pucb_intf__DOT__o_mem_wen));
    bufp->chgCData(oldp+58,(vlSelfRef.pucb_intf__DOT__o_mem_addr),4);
    bufp->chgBit(oldp+59,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_stall));
    bufp->chgBit(oldp+60,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_wen));
    bufp->chgBit(oldp+61,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_mshr));
    bufp->chgIData(oldp+62,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_atag),27);
    bufp->chgCData(oldp+63,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_set),5);
    bufp->chgCData(oldp+64,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_vtag),7);
    bufp->chgIData(oldp+65,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_field_table_wr_data),32);
    bufp->chgCData(oldp+66,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_field_table_vtag),7);
    bufp->chgCData(oldp+67,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_vtag),7);
    bufp->chgBit(oldp+68,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_validity_table_null));
    bufp->chgCData(oldp+69,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur),2);
    bufp->chgCData(oldp+70,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_cur_bits),3);
    bufp->chgCData(oldp+71,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt),2);
    bufp->chgCData(oldp+72,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_lru_nxt_bits),3);
    bufp->chgBit(oldp+73,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_miss));
    bufp->chgCData(oldp+74,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr),7);
    bufp->chgBit(oldp+75,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_rev_ptr_null));
    bufp->chgBit(oldp+76,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume));
    bufp->chgBit(oldp+77,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_consume));
    bufp->chgCData(oldp+78,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_cline),2);
    bufp->chgCData(oldp+79,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_ctag),4);
    bufp->chgCData(oldp+80,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_ctag),4);
    bufp->chgCData(oldp+81,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_vtp_offset),2);
    bufp->chgBit(oldp+82,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_w_cb_consume_sel));
    bufp->chgCData(oldp+83,(vlSelfRef.pucb_intf__DOT__dbg_pucb_intf_wo_cb_lru_ptag),4);
    bufp->chgBit(oldp+84,(vlSelfRef.pucb_intf__DOT__w_field_stall));
    bufp->chgBit(oldp+85,(vlSelfRef.pucb_intf__DOT__w_field_wen));
    bufp->chgIData(oldp+86,(vlSelfRef.pucb_intf__DOT__w_field_addr),32);
    bufp->chgBit(oldp+87,(vlSelfRef.pucb_intf__DOT__w_field_mshr));
    bufp->chgIData(oldp+88,(vlSelfRef.pucb_intf__DOT__w_field_atag),27);
    bufp->chgCData(oldp+89,(vlSelfRef.pucb_intf__DOT__w_field_set),5);
    bufp->chgCData(oldp+90,(vlSelfRef.pucb_intf__DOT__w_field_table_vtag),7);
    bufp->chgIData(oldp+91,(vlSelfRef.pucb_intf__DOT__w_field_table_wr_data),32);
    bufp->chgCData(oldp+92,(vlSelfRef.pucb_intf__DOT__wo_field_table_vtag),7);
    bufp->chgIData(oldp+93,(vlSelfRef.pucb_intf__DOT__wo_field_table_wr_data),32);
    bufp->chgCData(oldp+94,(vlSelfRef.pucb_intf__DOT__wo_validity_table_vtag),7);
    bufp->chgBit(oldp+95,(vlSelfRef.pucb_intf__DOT__wo_validity_table_null));
    bufp->chgCData(oldp+96,(vlSelfRef.pucb_intf__DOT__w_comp_hit),4);
    bufp->chgCData(oldp+97,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag))),4);
    bufp->chgCData(oldp+98,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                     >> 4U))),4);
    bufp->chgCData(oldp+99,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                     >> 8U))),4);
    bufp->chgCData(oldp+100,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_comp_ptag) 
                                      >> 0xcU))),4);
    bufp->chgCData(oldp+101,(vlSelfRef.pucb_intf__DOT__w_comp_lru_ptag),4);
    bufp->chgCData(oldp+102,(vlSelfRef.pucb_intf__DOT__w_comp_pvld),4);
    bufp->chgCData(oldp+103,(vlSelfRef.pucb_intf__DOT__w_cb_lru_cur),2);
    bufp->chgCData(oldp+104,(vlSelfRef.pucb_intf__DOT__w_cb_lru_cur_bits),3);
    bufp->chgCData(oldp+105,(vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt),2);
    bufp->chgCData(oldp+106,(vlSelfRef.pucb_intf__DOT__w_cb_lru_nxt_bits),3);
    bufp->chgBit(oldp+107,(vlSelfRef.pucb_intf__DOT__w_cb_miss));
    bufp->chgCData(oldp+108,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf))),4);
    bufp->chgCData(oldp+109,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__w_cb_consume_buf) 
                                      >> 4U))),4);
    bufp->chgCData(oldp+110,(vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr),7);
    bufp->chgBit(oldp+111,(vlSelfRef.pucb_intf__DOT__w_cb_rev_ptr_null));
    bufp->chgBit(oldp+112,(vlSelfRef.pucb_intf__DOT__w_cb_consume));
    bufp->chgBit(oldp+113,(vlSelfRef.pucb_intf__DOT__wo_cb_consume));
    bufp->chgCData(oldp+114,(vlSelfRef.pucb_intf__DOT__w_cb_cline),2);
    bufp->chgCData(oldp+115,(vlSelfRef.pucb_intf__DOT__w_cb_ctag),4);
    bufp->chgCData(oldp+116,(vlSelfRef.pucb_intf__DOT__wo_cb_ctag),4);
    bufp->chgCData(oldp+117,(vlSelfRef.pucb_intf__DOT__w_cb_vtp_offset),2);
    bufp->chgCData(oldp+118,(vlSelfRef.pucb_intf__DOT__wo_cb_vtp_offset),2);
    bufp->chgBit(oldp+119,(vlSelfRef.pucb_intf__DOT__w_cb_consume_sel));
    bufp->chgCData(oldp+120,(vlSelfRef.pucb_intf__DOT__wo_cb_lru_ptag),4);
    bufp->chgBit(oldp+121,(vlSelfRef.pucb_intf__DOT__w_mem_wen));
    bufp->chgCData(oldp+122,(vlSelfRef.pucb_intf__DOT__w_mem_addr),4);
    bufp->chgBit(oldp+123,(vlSelfRef.pucb_intf__DOT__r_field_mshr));
    bufp->chgIData(oldp+124,(vlSelfRef.pucb_intf__DOT__r_field_addr),32);
    bufp->chgBit(oldp+125,(vlSelfRef.pucb_intf__DOT__r_field_wen));
    bufp->chgCData(oldp+126,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag))),4);
    bufp->chgCData(oldp+127,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                      >> 4U))),4);
    bufp->chgCData(oldp+128,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                      >> 8U))),4);
    bufp->chgCData(oldp+129,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_ptag) 
                                      >> 0xcU))),4);
    bufp->chgCData(oldp+130,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_hit),4);
    bufp->chgCData(oldp+131,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_pvld),4);
    bufp->chgCData(oldp+132,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru),2);
    bufp->chgCData(oldp+133,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__i_field_lru_ptag),4);
    bufp->chgCData(oldp+134,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_line),2);
    bufp->chgCData(oldp+135,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__o_cb_ptag),4);
    bufp->chgCData(oldp+136,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru),2);
    bufp->chgCData(oldp+137,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_lru_ptag),4);
    bufp->chgCData(oldp+138,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_hit),4);
    bufp->chgCData(oldp+139,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_pvld),4);
    bufp->chgCData(oldp+140,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag))),4);
    bufp->chgCData(oldp+141,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                      >> 4U))),4);
    bufp->chgCData(oldp+142,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                      >> 8U))),4);
    bufp->chgCData(oldp+143,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_field_ptag) 
                                      >> 0xcU))),4);
    bufp->chgCData(oldp+144,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_line),2);
    bufp->chgCData(oldp+145,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__w_comp_ptag),4);
    bufp->chgIData(oldp+146,(vlSelfRef.pucb_intf__DOT__comparator_i__DOT__k),32);
    bufp->chgBit(oldp+147,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_clk));
    bufp->chgBit(oldp+148,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_rst));
    bufp->chgBit(oldp+149,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__i_cb_consume));
    bufp->chgBit(oldp+150,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_lru_sel));
    bufp->chgCData(oldp+151,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__o_cb_vtp_offset),2);
    bufp->chgBit(oldp+152,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_lru_sel));
    bufp->chgCData(oldp+153,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_nxt_cb_vtp_offset),2);
    bufp->chgBit(oldp+154,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_dec_stg_tap));
    bufp->chgBit(oldp+155,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__w_update_stg_tap));
    bufp->chgBit(oldp+156,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_dec_stg));
    bufp->chgCData(oldp+157,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_consume_update_stg),2);
    bufp->chgBit(oldp+158,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_lru_sel));
    bufp->chgCData(oldp+159,(vlSelfRef.pucb_intf__DOT__controller_i__DOT__r_cb_vtp_offset),2);
    bufp->chgBit(oldp+160,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_clk));
    bufp->chgBit(oldp+161,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rst));
    bufp->chgIData(oldp+162,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_atag),27);
    bufp->chgCData(oldp+163,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_field_lru),2);
    bufp->chgCData(oldp+164,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_rd_addr),5);
    bufp->chgCData(oldp+165,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_addr),7);
    bufp->chgIData(oldp+166,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_data),32);
    bufp->chgBit(oldp+167,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__i_wr_en));
    bufp->chgCData(oldp+168,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_hit),4);
    bufp->chgCData(oldp+169,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_lru_ptag),4);
    bufp->chgCData(oldp+170,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag))),4);
    bufp->chgCData(oldp+171,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                      >> 4U))),4);
    bufp->chgCData(oldp+172,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                      >> 8U))),4);
    bufp->chgCData(oldp+173,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__o_comp_ptag) 
                                      >> 0xcU))),4);
    bufp->chgCData(oldp+174,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_lru),2);
    bufp->chgIData(oldp+175,((0x7ffffffU & vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U])),27);
    bufp->chgIData(oldp+176,((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                             << 5U) 
                                            | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[0U] 
                                               >> 0x1bU)))),27);
    bufp->chgIData(oldp+177,((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                             << 0xaU) 
                                            | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[1U] 
                                               >> 0x16U)))),27);
    bufp->chgIData(oldp+178,((0x7ffffffU & ((vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[3U] 
                                             << 0xfU) 
                                            | (vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_atag[2U] 
                                               >> 0x11U)))),27);
    bufp->chgCData(oldp+179,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag))),4);
    bufp->chgCData(oldp+180,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                      >> 4U))),4);
    bufp->chgCData(oldp+181,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                      >> 8U))),4);
    bufp->chgCData(oldp+182,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_field_ptag) 
                                      >> 0xcU))),4);
    bufp->chgCData(oldp+183,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_tag_hit),4);
    bufp->chgIData(oldp+184,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__k),32);
    bufp->chgIData(oldp+185,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__m),32);
    bufp->chgCData(oldp+186,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_set),5);
    bufp->chgCData(oldp+187,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_wr_line),2);
    bufp->chgIData(oldp+188,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[0U]),32);
    bufp->chgIData(oldp+189,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[1U]),32);
    bufp->chgIData(oldp+190,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[2U]),32);
    bufp->chgIData(oldp+191,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_set_rd[3U]),32);
    bufp->chgIData(oldp+192,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wr),32);
    bufp->chgCData(oldp+193,(vlSelfRef.pucb_intf__DOT__field_table_i__DOT__w_line_wen),4);
    bufp->chgBit(oldp+194,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_clk));
    bufp->chgBit(oldp+195,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_rst));
    bufp->chgCData(oldp+196,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_vtag),7);
    bufp->chgIData(oldp+197,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_field_table_wr_data),32);
    bufp->chgBit(oldp+198,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_consume));
    bufp->chgCData(oldp+199,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_ctag),4);
    bufp->chgCData(oldp+200,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__i_cb_vtp_offset),2);
    bufp->chgCData(oldp+201,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_vtag),7);
    bufp->chgIData(oldp+202,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_field_table_wr_data),32);
    bufp->chgBit(oldp+203,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_consume));
    bufp->chgCData(oldp+204,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_ctag),4);
    bufp->chgCData(oldp+205,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_lru_ptag),4);
    bufp->chgCData(oldp+206,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__o_cb_vtp_offset),2);
    bufp->chgBit(oldp+207,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__w_en));
    bufp->chgCData(oldp+208,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__wo_cb_lru_ptag),4);
    bufp->chgCData(oldp+209,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cntr),2);
    bufp->chgCData(oldp+210,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_vtag),7);
    bufp->chgIData(oldp+211,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_field_table_wr_data),32);
    bufp->chgBit(oldp+212,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_consume));
    bufp->chgCData(oldp+213,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_ctag),4);
    bufp->chgCData(oldp+214,(vlSelfRef.pucb_intf__DOT__in_cdc_i__DOT__r_cb_vtp_offset),2);
    bufp->chgBit(oldp+215,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_clk));
    bufp->chgBit(oldp+216,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rst));
    bufp->chgCData(oldp+217,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_rd_addr),5);
    bufp->chgCData(oldp+218,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_wr_addr),5);
    bufp->chgCData(oldp+219,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt_bits),3);
    bufp->chgCData(oldp+220,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_nxt),2);
    bufp->chgBit(oldp+221,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_lru_wen));
    bufp->chgBit(oldp+222,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__i_field_stall));
    bufp->chgCData(oldp+223,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur_bits),3);
    bufp->chgCData(oldp+224,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__o_field_lru_cur),2);
    bufp->chgIData(oldp+225,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__k),32);
    bufp->chgCData(oldp+226,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_rd),5);
    bufp->chgCData(oldp+227,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wr),5);
    bufp->chgIData(oldp+228,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_lru_wen),32);
    bufp->chgBit(oldp+229,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__w_wr_vld));
    bufp->chgCData(oldp+230,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[0]),5);
    bufp->chgCData(oldp+231,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[1]),5);
    bufp->chgCData(oldp+232,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[2]),5);
    bufp->chgCData(oldp+233,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[3]),5);
    bufp->chgCData(oldp+234,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[4]),5);
    bufp->chgCData(oldp+235,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[5]),5);
    bufp->chgCData(oldp+236,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[6]),5);
    bufp->chgCData(oldp+237,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[7]),5);
    bufp->chgCData(oldp+238,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[8]),5);
    bufp->chgCData(oldp+239,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[9]),5);
    bufp->chgCData(oldp+240,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[10]),5);
    bufp->chgCData(oldp+241,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[11]),5);
    bufp->chgCData(oldp+242,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[12]),5);
    bufp->chgCData(oldp+243,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[13]),5);
    bufp->chgCData(oldp+244,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[14]),5);
    bufp->chgCData(oldp+245,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[15]),5);
    bufp->chgCData(oldp+246,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[16]),5);
    bufp->chgCData(oldp+247,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[17]),5);
    bufp->chgCData(oldp+248,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[18]),5);
    bufp->chgCData(oldp+249,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[19]),5);
    bufp->chgCData(oldp+250,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[20]),5);
    bufp->chgCData(oldp+251,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[21]),5);
    bufp->chgCData(oldp+252,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[22]),5);
    bufp->chgCData(oldp+253,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[23]),5);
    bufp->chgCData(oldp+254,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[24]),5);
    bufp->chgCData(oldp+255,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[25]),5);
    bufp->chgCData(oldp+256,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[26]),5);
    bufp->chgCData(oldp+257,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[27]),5);
    bufp->chgCData(oldp+258,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[28]),5);
    bufp->chgCData(oldp+259,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[29]),5);
    bufp->chgCData(oldp+260,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[30]),5);
    bufp->chgCData(oldp+261,(vlSelfRef.pucb_intf__DOT__lru_regfile_i__DOT__r_lru_state[31]),5);
    bufp->chgCData(oldp+262,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__i_field_lru_cur_bits),3);
    bufp->chgCData(oldp+263,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt_bits),3);
    bufp->chgCData(oldp+264,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__o_field_lru_nxt),2);
    bufp->chgIData(oldp+265,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__read_ptr__Vstatic__i),32);
    bufp->chgIData(oldp+266,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__read_ptr__Vstatic__k),32);
    bufp->chgCData(oldp+267,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_nactv),3);
    bufp->chgCData(oldp+268,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_tbits),3);
    bufp->chgCData(oldp+269,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_pactv),3);
    bufp->chgCData(oldp+270,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_cur_bits),3);
    bufp->chgCData(oldp+271,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt_bits),3);
    bufp->chgCData(oldp+272,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__w_field_lru_nxt),2);
    bufp->chgBit(oldp+273,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__w_tread));
    bufp->chgCData(oldp+274,(vlSelfRef.pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__w_tread),2);
    bufp->chgBit(oldp+275,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_clk));
    bufp->chgBit(oldp+276,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rst));
    bufp->chgCData(oldp+277,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_rd_addr),4);
    bufp->chgCData(oldp+278,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_addr),4);
    bufp->chgCData(oldp+279,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_wr_data),7);
    bufp->chgBit(oldp+280,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_evict));
    bufp->chgBit(oldp+281,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__i_field_consume));
    bufp->chgBit(oldp+282,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr_null));
    bufp->chgCData(oldp+283,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__o_rev_ptr),7);
    bufp->chgCData(oldp+284,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_addr),4);
    bufp->chgCData(oldp+285,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_addr),4);
    bufp->chgCData(oldp+286,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rd_data),8);
    bufp->chgBit(oldp+287,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr_null));
    bufp->chgCData(oldp+288,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_rev_ptr),7);
    bufp->chgCData(oldp+289,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_data),8);
    bufp->chgBit(oldp+290,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en_flag));
    bufp->chgSData(oldp+291,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__w_wr_en),16);
    bufp->chgCData(oldp+292,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[0]),8);
    bufp->chgCData(oldp+293,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[1]),8);
    bufp->chgCData(oldp+294,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[2]),8);
    bufp->chgCData(oldp+295,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[3]),8);
    bufp->chgCData(oldp+296,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[4]),8);
    bufp->chgCData(oldp+297,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[5]),8);
    bufp->chgCData(oldp+298,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[6]),8);
    bufp->chgCData(oldp+299,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[7]),8);
    bufp->chgCData(oldp+300,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[8]),8);
    bufp->chgCData(oldp+301,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[9]),8);
    bufp->chgCData(oldp+302,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[10]),8);
    bufp->chgCData(oldp+303,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[11]),8);
    bufp->chgCData(oldp+304,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[12]),8);
    bufp->chgCData(oldp+305,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[13]),8);
    bufp->chgCData(oldp+306,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[14]),8);
    bufp->chgCData(oldp+307,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__r_ptr_table[15]),8);
    bufp->chgIData(oldp+308,(vlSelfRef.pucb_intf__DOT__rev_ptr_table_i__DOT__k),32);
    bufp->chgBit(oldp+309,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_clk));
    bufp->chgBit(oldp+310,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rst));
    bufp->chgBit(oldp+311,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_stall));
    bufp->chgBit(oldp+312,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_mshr));
    bufp->chgCData(oldp+313,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_lru),2);
    bufp->chgBit(oldp+314,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_miss));
    bufp->chgCData(oldp+315,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_rd_addr),5);
    bufp->chgCData(oldp+316,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_addr),7);
    bufp->chgBit(oldp+317,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_wr_null));
    bufp->chgBit(oldp+318,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__i_field_consume));
    bufp->chgBit(oldp+319,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_consume));
    bufp->chgCData(oldp+320,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__o_field_pvld),4);
    bufp->chgCData(oldp+321,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_pvld),4);
    bufp->chgBit(oldp+322,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__wo_field_consume));
    bufp->chgCData(oldp+323,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_addr),7);
    bufp->chgCData(oldp+324,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_rd_data),4);
    bufp->chgBit(oldp+325,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_data));
    bufp->chgWData(oldp+326,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_en),128);
    bufp->chgBit(oldp+330,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__w_wr_null));
    bufp->chgWData(oldp+331,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__r_vld_table),128);
    bufp->chgIData(oldp+335,(vlSelfRef.pucb_intf__DOT__validity_table_i__DOT__k),32);
    bufp->chgBit(oldp+336,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_miss));
    bufp->chgBit(oldp+337,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ocb_consume));
    bufp->chgBit(oldp+338,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_wen));
    bufp->chgIData(oldp+339,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_atag),27);
    bufp->chgCData(oldp+340,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_field_set),5);
    bufp->chgCData(oldp+341,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_ctag),4);
    bufp->chgCData(oldp+342,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_cline),2);
    bufp->chgCData(oldp+343,((0xfU & (IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru))),4);
    bufp->chgCData(oldp+344,((0xfU & ((IData)(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru) 
                                      >> 4U))),4);
    bufp->chgBit(oldp+345,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_lru_sel));
    bufp->chgCData(oldp+346,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_ofield_table_vtag),7);
    bufp->chgBit(oldp+347,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr_null));
    bufp->chgCData(oldp+348,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__i_cb_rev_ptr),7);
    bufp->chgCData(oldp+349,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_vtag),7);
    bufp->chgIData(oldp+350,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_field_table_wr_data),32);
    bufp->chgBit(oldp+351,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_null));
    bufp->chgCData(oldp+352,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_ovalidity_table_vtag),7);
    bufp->chgBit(oldp+353,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_wen));
    bufp->chgCData(oldp+354,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__o_mem_addr),4);
    bufp->chgCData(oldp+355,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_vtag),7);
    bufp->chgIData(oldp+356,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_field_table_wr_data),32);
    bufp->chgCData(oldp+357,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_cb_lru_ptag),4);
    bufp->chgBit(oldp+358,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_null));
    bufp->chgCData(oldp+359,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__wo_validity_table_vtag),7);
    bufp->chgBit(oldp+360,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_wen));
    bufp->chgCData(oldp+361,(vlSelfRef.pucb_intf__DOT__wb_gen_i__DOT__w_mem_addr),4);
}

void Vtop___024root__trace_cleanup(void* voidSelf, VerilatedFst* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop___024root__trace_cleanup\n"); );
    // Init
    Vtop___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtop___024root*>(voidSelf);
    Vtop__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VlUnpacked<CData/*0:0*/, 1> __Vm_traceActivity;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        __Vm_traceActivity[__Vi0] = 0;
    }
    // Body
    vlSymsp->__Vm_activity = false;
    __Vm_traceActivity[0U] = 0U;
}
