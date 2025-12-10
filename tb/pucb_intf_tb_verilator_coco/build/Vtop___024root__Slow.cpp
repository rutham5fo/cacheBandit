// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop.h for the primary calling header

#include "Vtop__pch.h"
#include "Vtop__Syms.h"
#include "Vtop___024root.h"

// Parameter definitions for Vtop___024root
constexpr CData/*7:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__REG_NULL_CONST;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__PU_CLK_PERIOD;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__CB_CLK_PERIOD;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__SETS;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__ADDR_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__DATA_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__PAGE_SIZE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__PIPE_DEP;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__DEC_DEP;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__TABLE_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__SET_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__PTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__VTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__FULL_ATAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__VALID_ATAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__ATAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__OFFS_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__DEC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__LRU_NODES;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__comparator_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__comparator_i__DOT__PTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__comparator_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__NODES;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__STAGES;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__READ_START;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__0__KET____DOT__READ_LEN;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__READ_START;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt__BRA__1__KET____DOT__READ_LEN;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_bits__BRA__0__KET____DOT__PROOT;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_bits__BRA__0__KET____DOT__PSIDE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_bits__BRA__1__KET____DOT__PROOT;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_bits__BRA__1__KET____DOT__PSIDE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_bits__BRA__2__KET____DOT__PROOT;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_bits__BRA__2__KET____DOT__PSIDE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_setup__BRA__0__KET____DOT__NROOT;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_setup__BRA__0__KET____DOT__NSIDE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_setup__BRA__1__KET____DOT__NROOT;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_setup__BRA__1__KET____DOT__NSIDE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_setup__BRA__2__KET____DOT__NROOT;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__pucb_lru_i__DOT__gen_lru_nxt_setup__BRA__2__KET____DOT__NSIDE;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__SETS;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__DATA_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__ATAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__PTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__SET_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__field_table_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__validity_table_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__validity_table_i__DOT__SETS;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__validity_table_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__validity_table_i__DOT__SET_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__SETS;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__NODES;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__SET_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__WRITE_WIDTH;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__lru_regfile_i__DOT__READ_WIDTH;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__BLOCK_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__SETS;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__BLOCK_L;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__rev_ptr_table_i__DOT__SET_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__PU_CLK_PERIOD;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__CB_CLK_PERIOD;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__DATA_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__VTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__PTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__OFFS_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__CDC_RATIO;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__in_cdc_i__DOT__CNTR_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__DEC_DEP;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__ASSOC;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__SETS;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__DATA_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__PTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__VTAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__ATAG_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__ASSOC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__SET_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__wb_gen_i__DOT__DEC_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__controller_i__DOT__PIPE_DEP;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__controller_i__DOT__DEC_DEP;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__controller_i__DOT__OFFS_W;
constexpr IData/*31:0*/ Vtop___024root::pucb_intf__DOT__controller_i__DOT__DEC_W;


void Vtop___024root___ctor_var_reset(Vtop___024root* vlSelf);

Vtop___024root::Vtop___024root(Vtop__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop___024root___ctor_var_reset(this);
}

void Vtop___024root___configure_coverage(Vtop___024root* vlSelf, bool first);

void Vtop___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
    Vtop___024root___configure_coverage(this, first);
}

Vtop___024root::~Vtop___024root() {
}

// Coverage
void Vtop___024root::__vlCoverInsert(uint32_t* countp, bool enable, const char* filenamep, int lineno, int column,
    const char* hierp, const char* pagep, const char* commentp, const char* linescovp) {
    uint32_t* count32p = countp;
    static uint32_t fake_zero_count = 0;
    std::string fullhier = std::string{VerilatedModule::name()} + hierp;
    if (!fullhier.empty() && fullhier[0] == '.') fullhier = fullhier.substr(1);
    if (!enable) count32p = &fake_zero_count;
    *count32p = 0;
    VL_COVER_INSERT(vlSymsp->_vm_contextp__->coveragep(), VerilatedModule::name(), count32p,  "filename",filenamep,  "lineno",lineno,  "column",column,
        "hier",fullhier,  "page",pagep,  "comment",commentp,  (linescovp[0] ? "linescov" : ""), linescovp);
}

// Toggle Coverage
void Vtop___024root::__vlCoverToggleInsert(int begin, int end, bool ranged, uint32_t* countp, bool enable, const char* filenamep, int lineno, int column,
    const char* hierp, const char* pagep, const char* commentp) {
    int step = (end >= begin) ? 1 : -1;
    for (int i = begin; i != end + step; i += step) {
        for (int j = 0; j < 2; j++) {
            uint32_t* count32p = countp;
            static uint32_t fake_zero_count = 0;
            std::string fullhier = std::string{VerilatedModule::name()} + hierp;
            if (!fullhier.empty() && fullhier[0] == '.') fullhier = fullhier.substr(1);
            std::string commentWithIndex = commentp;
            if (ranged) commentWithIndex += '[' + std::to_string(i) + ']';
            commentWithIndex += j ? ":0->1" : ":1->0";
            if (!enable) count32p = &fake_zero_count;
            *count32p = 0;
            VL_COVER_INSERT(vlSymsp->_vm_contextp__->coveragep(), VerilatedModule::name(), count32p,  "filename",filenamep,  "lineno",lineno,  "column",column,
                "hier",fullhier,  "page",pagep,  "comment",commentWithIndex.c_str(),  "", "");
            ++countp;
        }
    }
}
