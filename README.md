

<h1> Fast Multi-Stack Slice-to-Volume Reconstruction via Multi-Scale Unrolled Optimization </h1>

<h2>Pre-requisites</h2>


<p>All pre-requisites are listed in <code>env_cSVR.yml</code>. The easiest way to install the environment is directly by downloading the environment file in
<a href="https://drive.google.com/drive/folders/14lG-uKZLcrR_jPe-SXfNCmOJfhzCV7mQ?usp=sharing" target="_blank">Google Drive</a>.</p>. Also, the last checkpoints of
the MLP and UNet model can be downloaded here.

<p>Use the following commands to set up the environment from .tar.gz file :</p>

<pre><code>
download from _env
mkdir -p ~/cSVR_env
tar -xzf cSVR_env.tar.gz -C ~/cSVR_env
source ~/cSVR_env/bin/activate
conda-unpack
</code></pre>

<p>Use the following commands to download the code base :</p>

<pre><code>
git clone https://github.com/mafirenze/cSVR.git
# Copy pretrained checkpoints into model_checkpoints folder

cp UNet_last.ckpt cSVR/model_checkpoints/
cp MLP_last.ckpt  cSVR/model_checkpoints/

</code></pre>

<h2> Usage </h2>

<h2>Input Data Format</h2>

<p>
This code expects <strong>three orthogonal image stacks per subject</strong>:
axial, sagittal, and coronal. Each stack must be provided as a NIfTI file
(<code>.nii</code> or <code>.nii.gz</code>) and follow the filename suffix
conventions listed below.
</p>

<ul>
  <li><code>_axi.nii.gz</code> — axial stack</li>
  <li><code>_sag.nii.gz</code> — sagittal stack</li>
  <li><code>_cor.nii.gz</code> — coronal stack</li>
</ul>

<p>
All stacks for a given subject must be spatially consistent and correspond
to the same anatomy.
</p>

<h2>Mask Files</h2>

<p>
If available, corresponding binary masks should be provided for each stack.
Mask filenames must match the stack orientation and be prefixed with
<code>mask_</code>:
</p>

<ul>
  <li><code>mask_axi.nii.gz</code></li>
  <li><code>mask_sag.nii.gz</code></li>
  <li><code>mask_cor.nii.gz</code></li>
</ul>

<p>
Masks must be defined in the same voxel space as their corresponding input
stacks.
</p>

<h2>Example Directory Structure</h2>

<pre><code>
sub0/
├── brain_axi.nii.gz
├── brain_sag.nii.gz
├── brain_cor.nii.gz
├── mask_axi.nii.gz
├── mask_sag.nii.gz
├── mask_cor.nii.gz

sub1/
├── brain_axi.nii.gz
├── brain_sag.nii.gz
├── brain_cor.nii.gz
├── mask_axi.nii.gz
├── mask_sag.nii.gz
├── mask_cor.nii.gz
</code></pre>

<h2>Mask Generation</h2>

<p>
If masks are <strong>not precomputed</strong>, they can be generated
automatically using:
</p>

<pre><code>
python get_masks.py --input-dir sub0 sub1
</code></pre>

<p>
This command creates orientation-specific masks and saves them in the
corresponding subject directories.
</p>

<h2>Reconstruction</h2>

<p>
Reconstruction can be performed using either
<strong>gradient descent–based optimization</strong> or an
<strong>implicit neural representation (INR)</strong> approach.
</p>

<h3>Gradient Descent Reconstruction</h3>

<pre><code>
python run_pipeline_cSVR.py sub0 sub1 \
    --suffix _run1.nii.gz \
    --run-cSVR \
    --gd-recon
</code></pre>

<h3>Implicit Neural Representation (INR) Reconstruction</h3>

<pre><code>
python run_pipeline_cSVR.py sub0 sub1 \
    --suffix _run1.nii.gz \
    --run-cSVR \
    --inr-recon
</code></pre>

<p>
The <code>--suffix</code> argument specifies a suffix appended to the output
reconstruction filenames.
</p>

<h2>Outputs</h2>

<ul>
  <li>
    <strong>Intermediate files</strong> (e.g., preprocessed slices,
    transforms, and optimization logs) are saved in:
    <pre><code>sub*/cSVR_files/</code></pre>
  </li>
  <li>
    <strong>Final reconstructed volumes</strong> are saved in the
    corresponding subject directory:
    <pre><code>sub*/</code></pre>
  </li>
</ul>

<h2>Notes</h2>

<ul>
  <li>Each subject must contain exactly one axial, one sagittal, and one coronal stack.</li>
  <li>Filenames must follow the specified suffix conventions to be detected correctly.</li>
  <li>Masks are optional but recommended for improved reconstruction quality.</li>
  <li>
    <strong>Bias field correction should be applied prior to reconstruction.</strong>
    We recommend using N4 bias field correction as implemented in
    <a href="https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html" target="_blank" rel="noopener noreferrer">
      SimpleITK (N4BiasFieldCorrection)
    </a>.
  </li>
</ul>
<h2>Work with us!</h2>

Feel free to reach out to me via e-mail for  questions, extensions, or applications!
