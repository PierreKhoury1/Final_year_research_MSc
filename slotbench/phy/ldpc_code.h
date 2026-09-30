// 5G NR LDPC (3GPP TS 38.212 section 5.3.2) on the host: lifting, compact edge lists for the GPU
// decoder, a GF(2) encoder for test codewords, and a reference layered min-sum decoder that runs
// the same per-check-node maths (phy_ops.h) as the GPU kernel.
//
// Lifting convention (as Sionna and 38.212): base entry V at (r, c) becomes a Z x Z shifted
// identity where lifted row r*Z + i connects to lifted column c*Z + (i + s) % Z, s = V mod Z.
// LLR convention everywhere: positive means bit 0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sb {

struct LdpcCode {
    int bg = 1;          // base graph 1 or 2
    int Z = 384;         // lifting size
    int i_ls = 1;        // lifting set index 0..7 (38.212 Table 5.3.2-1)
    int kb = 22;         // information base columns (22 for BG1, 10 for BG2)
    int rows = 46;       // base rows used (4..46 for BG1, 4..42 for BG2); fewer rows = higher rate
    int cols = 68;       // base columns used = kb + rows
    // compact per-row edge lists over the used rows/columns, rows in order 0..rows-1
    std::vector<int> row_start;   // size rows + 1
    std::vector<int> edge_col;    // base column of each edge
    std::vector<int> edge_shift;  // V mod Z of each edge
    int max_row_degree = 0;

    int n_edges() const { return (int)edge_col.size(); }
    int n_bits() const { return cols * Z; }          // codeword length incl. the 2Z punctured bits
    int k_bits() const { return kb * Z; }            // information bits
    int n_tx_bits() const { return (cols - 2) * Z; } // bits actually transmitted (first 2Z punctured)
    std::string describe() const;                    // one-line summary
};

// Lifting set index for Z (38.212 Table 5.3.2-1), or -1 if Z is not a valid lifting size.
int ldpc_set_index(int Z);

// Throws std::invalid_argument on a bad bg / Z / rows combination.
LdpcCode make_ldpc_code(int bg, int Z, int rows);

// info: k_bits() values in {0,1}. Returns n_bits() codeword with H * c = 0 over the used rows.
std::vector<uint8_t> ldpc_encode(const LdpcCode &code, const std::vector<uint8_t> &info);

// True when every used parity check is satisfied.
bool ldpc_check(const LdpcCode &code, const std::vector<uint8_t> &cw);

// Reference decoder: layered normalized min-sum, fixed iterations, alpha scales check messages.
// llr_in: n_bits() channel LLRs (punctured positions should be 0). bits_out: n_bits() hard decisions.
void ldpc_decode_host(const LdpcCode &code, const float *llr_in, int iters, float alpha, uint8_t *bits_out);

}  // namespace sb
