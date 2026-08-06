import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.stats import lognorm
from scipy.ndimage import distance_transform_edt
from skimage import io, color, morphology, measure
from skimage.segmentation import watershed as sk_watershed
from skimage.draw import polygon as sk_polygon
import imageio
import warnings
import tempfile

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= PERCORSI E SOGLIE =================
USER_HOME = os.path.expanduser("~")
SAVE_DIR = os.path.join(USER_HOME, "OneDrive", "Desktop", "segmentazione", "Crete")

try:
    os.makedirs(SAVE_DIR, exist_ok=True)
except PermissionError:
    fallback = os.path.join(tempfile.gettempdir(), "segmentazione_FERRET")
    os.makedirs(fallback, exist_ok=True)
    SAVE_DIR = fallback

# Aumentato a 15 per evitare di calcolare il rumore del sensore fotografico (prima era 2)
MIN_AREA_PX = 15
MAX_AREA_PX = 50000

# ============== CONFIGURAZIONE SAM ==============
SAM_CHECKPOINT = r"C:\Users\dario\PyCharmMiscProject\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

_mask_generator = None


def _init_sam(device=None):
    global _mask_generator
    if _mask_generator is not None:
        return _mask_generator

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"\n[ERRORE] Non ho trovato il file dei pesi di SAM (versione VIT_B) al percorso:\n'{SAM_CHECKPOINT}'"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Caricamento modello SAM ({SAM_MODEL_TYPE}) su device: {device}")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device)

    # [OTTIMIZZAZIONI VELOCITÀ]
    _mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # Ridotto da 64 a 32: 4 VOLTE PIÙ VELOCE, mantenendo super-dettaglio sulla Tile.
        points_per_batch=128,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=0,  # SPENTO: evita che SAM ritagli inutilmente la Tile che abbiamo già ritagliato noi.
        min_mask_region_area=MIN_AREA_PX,
    )
    return _mask_generator


def _tile_image(img, tile_size=1024, overlap=128):
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            y1, x1 = max(0, y2 - tile_size), max(0, x2 - tile_size)
            tiles.append((y1, y2, x1, x2))
    return tiles


def _local_mask_iou(m1, m2):
    """Calcola l'IoU velocemente proiettando solo l'area di intersezione dei due bounding box."""
    b1, b2 = m1["bbox"], m2["bbox"]
    ymin = max(b1[0], b2[0])
    xmin = max(b1[1], b2[1])
    ymax = min(b1[2], b2[2])
    xmax = min(b1[3], b2[3])

    # Exit istantaneo (costo computazionale ~ 0)
    if ymin > ymax or xmin > xmax:
        return 0.0

    # Operazioni bit a bit (bitwise &) su array booleani: più rapido di np.logical_and
    slice1 = m1["mask_local"][ymin - b1[0]:ymax - b1[0] + 1, xmin - b1[1]:xmax - b1[1] + 1]
    slice2 = m2["mask_local"][ymin - b2[0]:ymax - b2[0] + 1, xmin - b2[1]:xmax - b2[1] + 1]

    inter = (slice1 & slice2).sum()
    if inter == 0:
        return 0.0

    union = m1["area"] + m2["area"] - inter
    return inter / union


def calcola_moda(valori, bins=80):
    if len(valori) == 0:
        return 0.0
    counts, bin_edges = np.histogram(valori, bins=bins)
    max_idx = np.argmax(counts)
    return (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0


def register_clast(prop, gsd, overlay, stats_list, id_start):
    area_px = prop.area
    if area_px < MIN_AREA_PX or area_px > MAX_AREA_PX:
        return 0

    mask_obj = prop.image
    bbox = prop.bbox

    color_rand = np.random.randint(40, 255, 3)

    for c in range(3):
        overlay[bbox[0]:bbox[2], bbox[1]:bbox[3], c] = np.where(
            mask_obj,
            overlay[bbox[0]:bbox[2], bbox[1]:bbox[3], c] * 0.4 + color_rand[c] * 0.6,
            overlay[bbox[0]:bbox[2], bbox[1]:bbox[3], c]
        )

    area_m2 = area_px * (gsd ** 2)
    equiv_diameter_m = 2.0 * np.sqrt(area_m2 / np.pi)
    stats_list.append({
        "ID_Clasto": id_start,
        "Area_pixel": area_px,
        "Area_mq": float(area_m2),
        "Diametro_Equiv_m": float(equiv_diameter_m)
    })
    return 1


def select_gsd_and_roi(image_rgb):
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.imshow(image_rgb)
    ax1.set_title("1) Clicca 2 punti per il segmento GSD, poi premi INVIO")
    ax1.axis("off")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig1)
    if len(pts) < 2:
        raise SystemExit("GSD non valido.")
    (x1, y1), (x2, y2) = pts
    dist_px = np.hypot(x2 - x1, y2 - y1)
    valore_m = float(input("Inserisci lunghezza reale (metri): ").strip())
    gsd = valore_m / dist_px

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.imshow(image_rgb)
    ax2.set_title("2) Clicca i vertici del poligono ROI, poi premi INVIO")
    ax2.axis("off")
    roi_pts = plt.ginput(n=0, timeout=0)
    plt.close(fig2)
    if len(roi_pts) < 3:
        raise SystemExit("ROI non valida.")
    return gsd, np.array(roi_pts, dtype=int)


def vegetation_mask_robust(roi_rgb):
    r, g, b = [roi_rgb[:, :, i].astype(np.float32) for i in range(3)]
    exg = 2 * g - r - b
    sat = color.rgb2hsv(roi_rgb)[:, :, 1]
    veg = (exg > 22) & (sat > 0.25)
    return morphology.binary_closing(veg, morphology.disk(2))


def watershed_micro_clasti(roi_rgb_masked, roi_mask_valid, device=None):
    mask_generator = _init_sam(device)
    h, w = roi_rgb_masked.shape[:2]
    all_masks = []

    tiles = _tile_image(roi_rgb_masked, tile_size=1024, overlap=128)
    print(f"[INFO] Suddivisione in {len(tiles)} tile per l'elaborazione SAM...")

    for i, (y1, y2, x1, x2) in enumerate(tiles, start=1):
        tile_valid = roi_mask_valid[y1:y2, x1:x2]

        # Filtro istantaneo: se la tile non fa parte della ROI tracciata, saltala subito!
        if tile_valid.sum() < (MIN_AREA_PX * 2):
            continue

        tile = roi_rgb_masked[y1:y2, x1:x2]
        print(f"[INFO] Elaborazione tile {i}/{len(tiles)}...")
        results = mask_generator.generate(tile)

        for r in results:
            seg = r["segmentation"] & tile_valid
            area = seg.sum()
            if area < MIN_AREA_PX or area > MAX_AREA_PX:
                continue

            pos = np.argwhere(seg)
            if pos.size == 0:
                continue

            ymin, xmin = pos.min(axis=0)
            ymax, xmax = pos.max(axis=0)

            seg_local = seg[ymin:ymax + 1, xmin:xmax + 1].copy()
            g_bbox = (y1 + ymin, x1 + xmin, y1 + ymax, x1 + xmax)

            all_masks.append({
                "mask_local": seg_local,
                "bbox": g_bbox,
                "area": area
            })

    print(f"[INFO] Maschere grezze rilevate: {len(all_masks)}. Fusione duplicati ad alta efficienza in corso...")

    # Ordina dal sasso più grande al più piccolo per privilegiare le forme integre
    all_masks.sort(key=lambda x: x["area"], reverse=True)
    kept_masks = []

    # Eliminazione duplicati ultra rapida
    for m in all_masks:
        if not any(_local_mask_iou(m, km) > 0.6 for km in kept_masks):
            kept_masks.append(m)

    print(f"[INFO] Fusione completata. Maschere uniche rimaste: {len(kept_masks)}.")

    # Costruzione localizzata della label map finale
    labels_ws = np.zeros((h, w), dtype=np.int32)
    kept_masks.sort(key=lambda x: x["area"])
    for idx, m in enumerate(kept_masks, start=1):
        b = m["bbox"]
        sub_view = labels_ws[b[0]:b[2] + 1, b[1]:b[3] + 1]
        free = m["mask_local"] & (sub_view == 0)
        sub_view[free] = idx

    return labels_ws


if __name__ == "__main__":
    print("[INFO] Avvio analisi granulometrica ottimizzata...")

    img_path = r"C:\Users\dario\OneDrive\Desktop\CARTELLA DARIO\ELABORAZIONI\foto_djiVDA\CRETE SECHE\DJI_0798.JPG"
    img = io.imread(img_path)
    img_rgb = img[:, :, :3].copy() if img.ndim == 3 else np.stack([img, img, img], axis=-1)

    gsd, roi_poly = select_gsd_and_roi(img_rgb)

    rr, cc = sk_polygon(roi_poly[:, 1], roi_poly[:, 0], img_rgb.shape)
    roi_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    roi_mask[rr, cc] = True

    y0, y1 = int(roi_poly[:, 1].min()), int(roi_poly[:, 1].max())
    x0, x1 = int(roi_poly[:, 0].min()), int(roi_poly[:, 0].max())

    roi_cropped = img_rgb[y0:y1 + 1, x0:x1 + 1].copy()
    roi_mask_cropped = roi_mask[y0:y1 + 1, x0:x1 + 1]

    veg_mask = vegetation_mask_robust(roi_cropped) & roi_mask_cropped
    roi_mask_valida = roi_mask_cropped & (~veg_mask)

    roi_cropped_masked = roi_cropped.copy()
    roi_cropped_masked[~roi_mask_valida] = 0

    area_roi_m2 = np.sum(roi_mask_valida) * (gsd ** 2)
    overlay = roi_cropped.copy()
    stats = []
    id_counter = 1

    print("[INFO] Estrazione SAM e segmentazione in corso...")
    labels_ws = watershed_micro_clasti(roi_cropped_masked, roi_mask_valida)

    print("[INFO] Generazione della suddivisione artificiale nei vuoti...")
    bg_mask = roi_mask_valida & (labels_ws == 0)
    if bg_mask.sum() > 0:
        bg_indices = np.argwhere(bg_mask)
        num_bg_pixels = len(bg_indices)

        pixel_per_fake_stone = 35
        num_fake_stones = max(1, num_bg_pixels // pixel_per_fake_stone)

        rng = np.random.default_rng(seed=42)
        selected_indices = rng.choice(num_bg_pixels, size=num_fake_stones, replace=False)
        seeds_coords = bg_indices[selected_indices]

        fake_markers = np.zeros_like(labels_ws)
        start_idx = int(labels_ws.max() + 1)
        for idx, (r, c) in enumerate(seeds_coords, start=start_idx):
            fake_markers[r, c] = idx

        dist_field = distance_transform_edt(fake_markers == 0)
        fake_labels = sk_watershed(dist_field, markers=fake_markers, mask=bg_mask)

        labels_ws[bg_mask] = fake_labels[bg_mask]

    props = measure.regionprops(labels_ws)
    for prop in props:
        if register_clast(prop, gsd, overlay, stats, id_counter):
            id_counter += 1

    if stats:
        df_stats = pd.DataFrame(stats)
        diametri = df_stats['Diametro_Equiv_m'].values
        aree = df_stats['Area_mq'].values
        d_mean = float(np.mean(diametri))
        d_mode = float(calcola_moda(diametri, bins=80))
        a_mean = float(np.mean(aree))
        a_mode = float(calcola_moda(aree, bins=80))
        report_text = (
            f"--- REPORT STATISTICO GRANULOMETRICO ---\n\n"
            f"• Area Totale ROI (senza veg.): {area_roi_m2:.6f} m^2\n"
            f"• Numero Totale Elementi Rilevati: {len(stats)}\n"
            f"• Diametro Equivalente Medio: {d_mean:.6f} m\n"
            f"• Diametro Equivalente Modale: {d_mode:.6f} m\n"
            f"• Area Media dei Clasti: {a_mean:.6f} m^2\n"
            f"• Area Modale dei Clasti: {a_mode:.6f} m^2\n"
        )
        df_stats.to_csv(os.path.join(SAVE_DIR, "statistiche_clasti.csv"), sep=";", index=False)
    else:
        report_text = "Nessun micro-clasto rilevato."

    imageio.imwrite(os.path.join(SAVE_DIR, "roi_originale.png"), roi_cropped)
    imageio.imwrite(os.path.join(SAVE_DIR, "segmentazione_overlay.png"), overlay)

    fig = plt.figure(figsize=(22, 13))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.8, 1], wspace=0.15, hspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(roi_cropped)
    roi_locale = roi_poly - [x0, y0]
    poly_patch1 = plt.Polygon(roi_locale, edgecolor='red', facecolor='none', lw=1.0)
    ax1.add_patch(poly_patch1)
    ax1.set_title("1) Immagine Originale (Area ROI selezionata)", fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlay)
    poly_patch2 = plt.Polygon(roi_locale, edgecolor='red', facecolor='none', lw=1.0)
    ax2.add_patch(poly_patch2)
    ax2.set_title(f"2) Segmentazione Micro-Clasti Dettagliata ({len(stats)} unita')", fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0])
    if stats:
        counts, bins, _ = ax3.hist(diametri, bins=60, density=False, color='teal', edgecolor='black', alpha=0.6,
                                   label='Dati Rilevati')
        try:
            shape, loc, scale = lognorm.fit(diametri, floc=0)
            x_fit = np.linspace(min(diametri), max(diametri), 300)
            # Normalizza la PDF log-normale in base al numero totale di blocchi e alla larghezza dei bin
            bin_width = (bins[-1] - bins[0]) / len(bins)
            pdf_fit = lognorm.pdf(x_fit, shape, loc=loc, scale=scale)
            pdf_fit_scaled = pdf_fit * len(diametri) * bin_width
            ax3.plot(x_fit, pdf_fit_scaled, 'r-', lw=2.5, label='Modello Log-Normale')
        except Exception:
            pass
        ax3.set_title("Curva Granulometrica vs Fit Log-Normale", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Diametro Equivalente (m)")
        ax3.set_ylabel("Numero di Blocchi")
        ax3.grid(axis='y', alpha=0.3, ls='--')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.text(0.05, 0.95, report_text, fontsize=11, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.6))

    output_dashboard_path = os.path.join(SAVE_DIR, "dashboard_analisi_finale.png")
    plt.savefig(output_dashboard_path, dpi=250, bbox_inches='tight')
    plt.show()

    print(f"[OK] Mappatura completata con successo. Identificati {len(stats)} elementi totali nella ROI.")