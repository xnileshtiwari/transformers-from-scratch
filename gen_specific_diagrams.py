import subprocess
import os

assets_dir = "/home/nilesh/code/transformers-from-scratch/assets"
banana_script = "/home/nilesh/.openclaw/skills/nano-banana/banana.py"

diagrams = {
    "diagram_0_2_embedding.jpg": (
        "A clear, textbook technical diagram of Transformer Token Embedding Lookup. "
        "Show a token integer ID (e.g. 4125 'quantum') converted into a one-hot vector. "
        "The vector multiplies a large 2D embedding weight matrix W_E of size V x d_model, extracting a single continuous row vector of size d_model. "
        "The extracted vector flows into the persistent Residual Stream. High contrast, clean educational illustration, white background."
    ),
    "diagram_1_1_permutation.jpg": (
        "An educational architecture comparison diagram of Recurrent Sequential Processing versus Transformer Permutation Invariance. "
        "On the left, an RNN processes words sequentially with directed state arrows h_1 -> h_2 -> h_3. "
        "On the right, a Transformer processes all tokens in parallel as an unordered set with all-to-all attention links, showing permutation equivariance. "
        "Clean, technical textbook style, white background."
    ),
    "diagram_1_2_sinusoidal.jpg": (
        "A technical diagram illustrating the Frequency Spectrum of Sinusoidal Positional Encoding in Transformers. "
        "Show dimension index on the vertical axis and token position on the horizontal axis. "
        "Show high-frequency sine waves with short wavelengths at low dimension indices (i=0) that tick rapidly, "
        "and low-frequency sine waves with long wavelengths at high dimension indices that tick slowly. Clean educational chart, white background."
    ),
    "diagram_2_1_projections.jpg": (
        "A clean technical diagram of Transformer Linear Projections for Self-Attention. "
        "Show input token matrix X of size N x d_model branching into three distinct linear weight matrices: W_Q, W_K, and W_V. "
        "Show the outputs labeled clearly: Query matrix Q [N x d_k], Key matrix K [N x d_k], and Value matrix V [N x d_v]. "
        "High contrast, minimalist educational style, white background."
    ),
    "diagram_2_2_dot_product.jpg": (
        "A technical diagram showing the Raw Compatibility Scoring matrix multiplication Q x K^T in self-attention. "
        "Show Query matrix Q of shape N x d_k multiplied by transposed Key matrix K^T of shape d_k x N, "
        "resulting in an N x N square matrix of raw pairwise compatibility scores S. Clean vector illustration, white background."
    ),
    "diagram_2_3_scaling.jpg": (
        "A technical diagram illustrating the Variance Scaling Factor 1/sqrt(d_k) in Transformers. "
        "Show an unscaled bell curve with high variance (+-35) pushing into flat saturation zones where gradients vanish. "
        "Beside it, show the division by sqrt(d_k) compressing the distribution to unit variance (+-1), keeping scores in the steep, active gradient region of Softmax. "
        "Clean educational chart, white background."
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

