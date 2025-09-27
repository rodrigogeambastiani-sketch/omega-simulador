# app.py
"""
ÔMEGA - Simulador TC (protótipo Streamlit) - baseado em GE 16 canais (protótipo)
Requer: streamlit, numpy, scipy, scikit-image, matplotlib, pydicom (opcional)
Rodar: streamlit run app.py
"""

import streamlit as st
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import io
from PIL import Image
import datetime
import os

# Optional: DICOM export
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

st.set_page_config(layout="wide", page_title="Ômega - Simulador TC (GE 16 canais)")

st.title("Ômega — Simulador de Tomografia (base: GE 16 canais)")

# Sidebar: parâmetros do scanner
st.sidebar.header("Parâmetros do Scanner (simulado GE 16ch)")
detector_rows = st.sidebar.number_input("Número de canais (detector rows)", value=16, min_value=1, max_value=64, step=1)
detector_elements = st.sidebar.number_input("Detectores por corte (amostras detectorais)", value=512, min_value=64, max_value=2048, step=1)
slice_thickness_mm = st.sidebar.number_input("Espessura de fatia (mm)", value=1.25, format="%.2f")
num_projections = st.sidebar.slider("Número de projeções (ângulos)", min_value=60, max_value=2048, value=360, step=10)
noise_level = st.sidebar.slider("Ruído (sigma gaussiano % do sinal)", 0.0, 100.0, 2.0) # percentual
apply_detector_nonuniformity = st.sidebar.checkbox("Simular não uniformidade do detector", value=True)
pitch = st.sidebar.slider("Pitch (simulado)", 0.1, 2.0, 1.0, 0.05)  # simplificação para helical
geometry_mode = st.sidebar.selectbox("Geometria", ["Paralela (Radon)", "Fan-beam (aprox.)"], index=1)

st.sidebar.markdown("---")
st.sidebar.header("Phantom / Corte")
phantom_size = st.sidebar.selectbox("Tamanho do phantom (pixels)", [128, 256, 512], index=1)
phantom_type = st.sidebar.selectbox("Phantom", ["Shepp-Logan (padrão)", "Custom (anel/objetos)"], index=0)
num_slices = st.sidebar.number_input("Número de fatias simuladas (stack)", value=16, min_value=1, max_value=64, step=1)

# Controls
st.sidebar.markdown("---")
if st.sidebar.button("Gerar simulação"):
    run_sim = True
else:
    run_sim = False

st.markdown("### Visualização e controles rápidos")
cols = st.columns([1,1,1,1])
with cols[0]:
    st.metric("Canal (rows)", f"{detector_rows}")
with cols[1]:
    st.metric("Detectores / corte", f"{detector_elements}")
with cols[2]:
    st.metric("Projeções", f"{num_projections}")
with cols[3]:
    st.metric("Fatias (stack)", f"{num_slices}")

# --- Utility functions
def build_phantom(size, kind="shepp_logan", slice_index=0, total_slices=1):
    """Return a float32 phantom image sized (size,size). If multi-slice, add small z variation."""
    if kind == "shepp_logan":
        base = shepp_logan_phantom()
        base = resize(base, (size, size), mode='reflect', anti_aliasing=True)
        # simulate slight variation across slices
        if total_slices > 1:
            variation = 0.05 * np.sin(2*np.pi*slice_index/total_slices)
            base = base * (1.0 + variation)
        return base.astype(np.float32)
    else:
        # simple custom: concentric rings
        x = np.linspace(-1,1,size)
        xv, yv = np.meshgrid(x,x)
        r = np.sqrt(xv**2 + yv**2)
        img = np.zeros_like(r)
        img += (r < 0.8).astype(float)
        img += 0.5 * (r < 0.4).astype(float)
        if total_slices > 1:
            img = gaussian_filter(img, sigma=0.5*(1+slice_index/total_slices))
        return img.astype(np.float32)

def simulate_sinogram(img, n_projections, detector_count, geometry="fan"):
    # Using Radon for parallel; for fan-beam we still use radon as approximation
    theta = np.linspace(0., 180., n_projections, endpoint=False)
    sino = radon(img, theta=theta, circle=True)
    # resize sinogram to detector_count (interpolate columns)
    from skimage.transform import resize
    sino_resized = resize(sino, (detector_count, n_projections), mode='reflect', anti_aliasing=True)
    return sino_resized, theta

def add_noise(sino, sigma_percent=2.0, nonuniform=False):
    scale = np.mean(np.abs(sino)) + 1e-9
    sigma = sigma_percent/100.0 * scale
    noisy = sino + np.random.normal(0, sigma, size=sino.shape)
    if nonuniform:
        # multiplicative gain variations across detectors
        gains = 1.0 + 0.05 * np.sin(np.linspace(0, 4*np.pi, sino.shape[0]))  # simple pattern
        noisy = noisy * gains[:,None]
    return noisy

def reconstruct_fbp(sino, theta, filter_name='ramp'):
    # sino shape expected (detector, angles) -> for iradon, need shape (Nangles, Ndetector) so transpose
    from skimage.transform import iradon
    sino_t = sino.T
    # iradon expects projections along columns (n_angles x n_detectors) and returns an image
    # We'll resample to square size
    out = iradon(sino_t, theta=theta, filter_name=filter_name, circle=True)
    return out

# --- Run simulation
if run_sim:
    st.info("Executando simulação... gerando stack de fatias e reconstruções (FBP).")
    phantoms = []
    sinograms = []
    noisy_sinos = []
    reconstructions = []

    # Loop over slices to simulate detector_rows channels (stack)
    for s in range(num_slices):
        img = build_phantom(phantom_size, kind=("shepp_logan" if phantom_type=="Shepp-Logan (padrão)" else "custom"), slice_index=s, total_slices=num_slices)
        phantoms.append(img)
        sino, theta = simulate_sinogram(img, num_projections, detector_elements, geometry=geometry_mode)
        sinograms.append(sino)
        noisy = add_noise(sino, sigma_percent=noise_level, nonuniform=apply_detector_nonuniformity)
        noisy_sinos.append(noisy)
        recon = reconstruct_fbp(noisy, theta, filter_name='ramp')
        reconstructions.append(recon)

    # Stack results
    phantoms_stack = np.stack(phantoms, axis=0)
    sinograms_stack = np.stack(sinograms, axis=0)
    noisy_stack = np.stack(noisy_sinos, axis=0)
    recon_stack = np.stack(reconstructions, axis=0)

    # Display first slice and options
    st.subheader("Resultados (slice 0 por padrão)")
    slice_idx = st.slider("Escolher fatia para visualizar", 0, num_slices-1, 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Phantom (entrada)**")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(phantoms_stack[slice_idx], cmap='gray')
        ax.set_axis_off()
        st.pyplot(fig)
    with col2:
        st.markdown("**Sinograma (aprox.)**")
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.imshow(noisy_stack[slice_idx], aspect='auto')
        ax2.set_xlabel("Ângulo")
        ax2.set_ylabel("Detector")
        st.pyplot(fig2)
    with col3:
        st.markdown("**Reconstrução (FBP)**")
        fig3, ax3 = plt.subplots(figsize=(3,3))
        ax3.imshow(recon_stack[slice_idx], cmap='gray')
        ax3.set_axis_off()
        st.pyplot(fig3)

    # Global visualizations
    st.markdown("### Visualizações 3D / Agregadas (MIP, média)")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("MIP (projeção máxima) do phantom")
        mip = np.max(phantoms_stack, axis=0)
        fig4, ax4 = plt.subplots(figsize=(4,4))
        ax4.imshow(mip, cmap='gray')
        ax4.set_axis_off()
        st.pyplot(fig4)
    with colB:
        st.markdown("Média das reconstruções")
        mean_recon = np.mean(recon_stack, axis=0)
        fig5, ax5 = plt.subplots(figsize=(4,4))
        ax5.imshow(mean_recon, cmap='gray')
        ax5.set_axis_off()
        st.pyplot(fig5)

    # Export buttons
    st.markdown("---")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if st.button("Salvar reconstrução atual como PNG"):
            buf = io.BytesIO()
            im = Image.fromarray(np.uint8(255*(recon_stack[slice_idx]/np.max(recon_stack[slice_idx]+1e-9))))
            im.save(buf, format="PNG")
            st.download_button("Download PNG", buf.getvalue(), file_name=f"recon_slice{slice_idx}.png", mime="image/png")
    with exp_col2:
        if HAVE_PYDICOM:
            if st.button("Exportar fatia atual como DICOM (simples)"):
                # Create simple DICOM file
                file_meta = pydicom.Dataset()
                ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
                ds.PatientName = "Simulado^Omega"
                ds.PatientID = "OMEGA001"
                ds.Modality = "CT"
                ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
                ds.Rows, ds.Columns = recon_stack[slice_idx].shape
                pixel_array = np.uint16( (recon_stack[slice_idx] - recon_stack[slice_idx].min()) / (recon_stack[slice_idx].ptp()+1e-9) * 65535 )
                ds.PixelData = pixel_array.tobytes()
                ds.save_as(f"recon_slice{slice_idx}.dcm")
                with open(f"recon_slice{slice_idx}.dcm","rb") as f:
                    st.download_button("Download DICOM", f.read(), file_name=f"recon_slice{slice_idx}.dcm", mime="application/dicom")
        else:
            st.info("pydicom não instalado — instale com `pip install pydicom` para habilitar export DICOM.")

    st.success("Simulação concluída. Ajuste parâmetros e gere novamente conforme necessário.")
else:
    st.info("Configure parâmetros na barra lateral e clique em 'Gerar simulação'.")

st.markdown("---")
st.markdown("### Notas técnicas (rápidas)")
st.markdown("""
- Este protótipo usa *Radon* / *Iradon* do scikit-image (implementação paralela) como aproximação.  
- O GE 16 canais real usa geometria fan-beam multi-slice e aquisição helicoidal — isto aqui é um protótipo 2D empilhado para treino prático.  
- Para evolução: adicionar geometria fan-beam explícita, correção de beam hardening, simulação de contraste iodado, movimento, artefatos metal, e modelo físico do colimador / detetores.
""")

st.markdown("### Instalação (ambiente Python recomendado)")
st.code("""
python -m venv venv
source venv/bin/activate    # mac/linux
venv\\Scripts\\activate     # windows
pip install streamlit numpy scipy scikit-image matplotlib pillow
# opcional: pydicom
pip install pydicom
streamlit run app.py
""")
