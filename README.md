# cacheBandit
Dynamic Cache framework for eleminating block/line conflicts in streaming applications.

## Introduction
Cache block conflicts can arise within a set due to limited associativity of cache. In other words, certain applications face cache conflicts
not due to cache capacity, but rather due to the thrashing of cache lines arising from the applications access pattern.
CacheBandit framework aims to reduce the number of external DRAM access by eliminating cache thrashing due to a cache's limited associativity.

## Project Structure and Setup
The project is still under construction but the simulation of cacheBandit and related metrics can be found in the sims and docs folders,
respectively.

cacheBandit\
├───docs\
└───sims\
	├───context\
	├───logs\
	└───spmat\

Run 'suitsparse.py' script to obtain sparse matrices from the 'SuitSparse Matrix Collection'.
Run 'stream_gen.py' to generate a memory trace (context) from the 'CSR' format of respective matrices.
Run 'cacheBandit.py' to simulate the behaviour of cache for the selected memory traces.

## Future plans
CacheBandit will be realized in hardware using parts from 'QuickPageLIte' project. The goal is to support single cycle latency operation 
to facilitate dynamic cache behaviour. The framework will be scalable for various frequencies by supporting parallel processing of allocation 
and deallocation requests from the stream. CacheBandit itself will run @ 100 Mhz, but can be scaled to support upto 800 Mhz streams by 
following 'Little's Law' to keep multiple concurrent works in progress.

	Little's Law => Works in Progress = Latency * IPC

For a stream operating @ 400 MHz, will see a latency of 4 cycles, hence cacheBandit will scale accordingly to support 4 concurrent 
requests per cycle @ 100 MHz.

