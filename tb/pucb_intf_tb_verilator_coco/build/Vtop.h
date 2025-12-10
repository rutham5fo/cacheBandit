// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary model header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef VERILATED_VTOP_H_
#define VERILATED_VTOP_H_  // guard

#include "verilated.h"
#include "verilated_cov.h"
#include "svdpi.h"

class Vtop__Syms;
class Vtop___024root;
class VerilatedFstC;

// This class is the main interface to the Verilated model
class alignas(VL_CACHE_LINE_BYTES) Vtop VL_NOT_FINAL : public VerilatedModel {
  private:
    // Symbol table holding complete model state (owned by this class)
    Vtop__Syms* const vlSymsp;

  public:

    // CONSTEXPR CAPABILITIES
    // Verilated with --trace?
    static constexpr bool traceCapable = true;

    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(&i_pu_clk,0,0);
    VL_IN8(&i_cb_clk,0,0);
    VL_IN8(&i_rst,0,0);
    VL_IN8(&i_cb_en,0,0);
    VL_IN8(&i_field_mshr,0,0);
    VL_IN8(&i_field_wen,0,0);
    VL_IN8(&i_cb_consume_buf,7,0);
    VL_OUT8(&o_field_mshr,0,0);
    VL_OUT8(&o_field_wen,0,0);
    VL_OUT8(&o_cb_consume,0,0);
    VL_OUT8(&o_cb_ptag,3,0);
    VL_OUT8(&o_cb_vtp_offset,1,0);
    VL_OUT8(&o_mem_wen,0,0);
    VL_OUT8(&o_mem_addr,3,0);
    VL_OUT8(&dbg_pucb_intf_w_field_stall,0,0);
    VL_OUT8(&dbg_pucb_intf_w_field_wen,0,0);
    VL_OUT8(&dbg_pucb_intf_w_field_mshr,0,0);
    VL_OUT8(&dbg_pucb_intf_w_field_set,4,0);
    VL_OUT8(&dbg_pucb_intf_w_field_table_vtag,6,0);
    VL_OUT8(&dbg_pucb_intf_wo_field_table_vtag,6,0);
    VL_OUT8(&dbg_pucb_intf_wo_validity_table_vtag,6,0);
    VL_OUT8(&dbg_pucb_intf_wo_validity_table_null,0,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_lru_cur,1,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_lru_cur_bits,2,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_lru_nxt,1,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_lru_nxt_bits,2,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_miss,0,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_rev_ptr,6,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_rev_ptr_null,0,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_consume,0,0);
    VL_OUT8(&dbg_pucb_intf_wo_cb_consume,0,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_cline,1,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_ctag,3,0);
    VL_OUT8(&dbg_pucb_intf_wo_cb_ctag,3,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_vtp_offset,1,0);
    VL_OUT8(&dbg_pucb_intf_w_cb_consume_sel,0,0);
    VL_OUT8(&dbg_pucb_intf_wo_cb_lru_ptag,3,0);
    VL_IN(&i_field_addr,31,0);
    VL_OUT(&o_field_addr,31,0);
    VL_OUT(&dbg_pucb_intf_w_field_atag,26,0);
    VL_OUT(&dbg_pucb_intf_w_field_table_wr_data,31,0);

    // CELLS
    // Public to allow access to /* verilator public */ items.
    // Otherwise the application code can consider these internals.

    // Root instance pointer to allow access to model internals,
    // including inlined /* verilator public_flat_* */ items.
    Vtop___024root* const rootp;

    // CONSTRUCTORS
    /// Construct the model; called by application code
    /// If contextp is null, then the model will use the default global context
    /// If name is "", then makes a wrapper with a
    /// single model invisible with respect to DPI scope names.
    explicit Vtop(VerilatedContext* contextp, const char* name = "TOP");
    explicit Vtop(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    virtual ~Vtop();
  private:
    VL_UNCOPYABLE(Vtop);  ///< Copying not allowed

  public:
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    /// Are there scheduled events to handle?
    bool eventsPending();
    /// Returns time at next time slot. Aborts if !eventsPending()
    uint64_t nextTimeSlot();
    /// Trace signals in the model; called by application code
    void trace(VerilatedTraceBaseC* tfp, int levels, int options = 0) { contextp()->trace(tfp, levels, options); }
    /// Retrieve name of this model instance (as passed to constructor).
    const char* name() const;

    // Abstract methods from VerilatedModel
    const char* hierName() const override final;
    const char* modelName() const override final;
    unsigned threads() const override final;
    /// Prepare for cloning the model at the process level (e.g. fork in Linux)
    /// Release necessary resources. Called before cloning.
    void prepareClone() const;
    /// Re-init after cloning the model at the process level (e.g. fork in Linux)
    /// Re-allocate necessary resources. Called after cloning.
    void atClone() const;
    std::unique_ptr<VerilatedTraceConfig> traceConfig() const override final;
  private:
    // Internal functions - trace registration
    void traceBaseModel(VerilatedTraceBaseC* tfp, int levels, int options);
};

#endif  // guard
