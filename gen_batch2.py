import subprocess
import os

assets_dir = "/home/nilesh/.openclaw/workspace/assets"
banana_script = "/home/nilesh/.openclaw/skills/nano-banana/banana.py"

diagrams = {
    "diagram_2_4_softmax.jpg": (
        "A clean technical diagram of Softmax row-wise probability normalization in self-attention. "
        "Show a row of raw scaled logits e.g. [1.2, 3.8, -0.5, 2.1]. "
        "Show the numeric steps: subtract maximum (3.8) for numerical stability, exponentiate with e^x, "
        "and divide by the sum of exponents to produce a strict probability distribution [0.06, 0.79, 0.01, 0.14] summing to 1.0. "
        "Textbook engineering style, clean lines, white background."
    ),
    "diagram_2_5_value_aggregation.jpg": (
        "A technical diagram of Weighted Value Aggregation in self-attention. "
        "Show an N x N square attention weight matrix A multiplying an N x d_v Value matrix V. "
        "Highlight that each output row z_i is a weighted sum (linear combination) of value vectors v_1, v_2, ..., v_N weighted by the attention probabilities A_i,j. "
        "Clean, high contrast, white background."
    ),
    "diagram_3_1_multihead.jpg": (
        "An educational architecture diagram showing Multi-Head Attention Subspace Splitting. "
        "Show input representation matrix X branching into h parallel attention heads. "
        "Each head has its own W_i^Q, W_i^K, W_i^V and computes attention in a smaller subspace of dimension d_k = d_model / h. "
        "Show the outputs labeled Head 1, Head 2, up to Head h. Minimalist, clear lines, white background."
    ),
    "diagram_3_2_output_proj.jpg": (
        "A technical architecture diagram showing the Multi-Head Output Projection W_O. "
        "Show multiple attention head output matrices (Head 1, Head 2, ... Head h) lined up horizontally and concatenated into a single wide matrix H_concat of width d_model. "
        "Show H_concat multiplied by a learned linear projection matrix W_O of size d_model x d_model to produce the final attention layer output. "
        "Clean engineering illustration, white background."
    ),
    "diagram_4_2_rmsnorm.jpg": (
        "A side-by-side comparison diagram of Layer Normalization versus RMSNorm in deep learning. "
        "On the left, LayerNorm shows subtracting the mean mu, dividing by standard deviation sigma, and scaling by gamma plus shift beta. "
        "On the right, RMSNorm shows dividing directly by the root-mean-square RMS(x) without calculating mean or beta, saving 50 percent of GPU memory operations. "
        "Clean textbook comparison chart, white background."
    ),
    "diagram_4_3_pre_post_ln.jpg": (
        "An architectural comparison diagram of Post-LN versus Pre-LN in Transformers. "
        "On the left, Post-LN (2017) shows the normalization block sitting directly on the residual stream after the addition, choking gradient flow. "
        "On the right, Pre-LN (modern LLMs) shows the clean residual stream highway passing straight through, with normalization only on the sublayer branch before attention. "
        "Textbook illustration, high contrast, white background."
    ),
    "diagram_5_1_ffn_expansion.jpg": (
        "A technical diagram of the Two-Layer Feed-Forward Network (FFN) Expansion in Transformers. "
        "Show input vector x of dimension d_model entering linear expansion matrix W_1, blowing up into a wide hidden layer of dimension 4*d_model acting as key concept detectors. "
        "Show activation function sigma firing, followed by linear contraction matrix W_2 projecting back down to d_model. "
        "Clean educational engineering diagram, white background."
    ),
    "diagram_7_3_dpo.jpg": (
        "A technical flowchart of Direct Preference Optimization (DPO). "
        "Show prompt x with two candidate responses: preferred y_w and dispreferred y_l. "
        "Show the calculation of implicit rewards using log probability ratios between policy model pi_theta and frozen reference model pi_ref. "
        "Show the delta between the winning ratio and losing ratio fed into a sigmoid binary cross-entropy loss, bypassing reward models completely. "
        "Clean textbook diagram, white background."
    )
}

for name, prompt in diagrams.items():
    out = os.path.join(assets_dir, name)
    if os.path.exists(out):
        print(f"Skipping {name}, exists.")
        continue
    print(f"Generating {name}...")
    cmd = ["python3", banana_script, prompt, "-o", out, "-s", "1K", "-a", "16:9"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Done: {name}")
    else:
        print(f"Failed {name}: {res.stderr}")
