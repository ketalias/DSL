"""
Лабораторна робота 1. Видалення шуму за його шаблоном.

Сценарій: запис лекції (мій голос), на який у кількох місцях
накладається клацання ручки з різною гучністю.

Запуск:
    python lab1.py [signal.wav] [noise.wav]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import wavfile
from scipy.signal import correlate, find_peaks, resample_poly

SEED = 42                 # для відтворюваності
POSITIONS_SEC = [2, 7, 12, 18, 24]  # де вставляємо шаблон, секунди
N_OCCURRENCES = len(POSITIONS_SEC)  # кількість входжень шаблону (>= 3)
TEMPLATE_NOISE_SNR_DB = 20  # SNR шаблону відносно доданого гаусівського шуму
DETECT_Z = 10             # поріг виявлення: у скільки робастних σ пік вищий за фон кореляції
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # шляхи — від теки скрипта, а не від поточної
OUT_DIR = os.path.join(BASE_DIR, "results")


# ---------------------------------------------------------------- 1. Завантаження
def load_audio(path, target_sr=None):
    """Завантажує аудіо у моно float [-1, 1]. Не-WAV конвертує через ffmpeg (або afconvert на macOS)."""
    if not path.lower().endswith(".wav"):
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # на Windows відкритий файл не дасть конвертеру в нього писати
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-acodec", "pcm_s16le", tmp]
        elif shutil.which("afconvert"):
            cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", path, tmp]
        else:
            sys.exit(f"Файл {path} не у форматі WAV. Встановіть ffmpeg або сконвертуйте його у .wav вручну.")
        subprocess.run(cmd, check=True)
        path = tmp
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if target_sr and sr != target_sr:
        x = resample_poly(x, target_sr, sr)
        sr = target_sr
    return sr, x


def trim_silence(x, sr, frame_ms=10, rel_threshold=0.05, pad_ms=20):
    """Обрізає тишу на початку та в кінці (за енергією кадрів)."""
    fr = int(sr * frame_ms / 1000)
    energy = np.array([np.sqrt(np.mean(x[i:i + fr] ** 2)) for i in range(0, len(x) - fr, fr)])
    active = np.where(energy > rel_threshold * energy.max())[0]
    pad = int(sr * pad_ms / 1000)
    start = max(active[0] * fr - pad, 0)
    end = min((active[-1] + 1) * fr + pad, len(x))
    return x[start:end]


def snr_db(clean, test):
    return 10 * np.log10(np.sum(clean ** 2) / np.sum((clean - test) ** 2))


def main():
    # Консоль Windows за замовчуванням не UTF-8 — без цього кирилиця у print() може впасти з помилкою
    sys.stdout.reconfigure(encoding="utf-8")
    signal_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "audio", "signal.wav")
    noise_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE_DIR, "audio", "noise.wav")
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sr, x = load_audio(signal_path)
    duration = len(x) / sr
    assert duration >= 5, f"Сигнал має бути не коротшим за 5 с (зараз {duration:.2f} с)"
    assert np.max(np.abs(x)) > 1e-3, "Сигнал порожній"
    print(f"Корисний сигнал: {duration:.2f} с, {sr} Гц")

    # ------------------------------------------------------------ 2. Шаблон шуму
    _, noise_raw = load_audio(noise_path, target_sr=sr)
    noise_sound = trim_silence(noise_raw, sr)
    noise_sound /= np.max(np.abs(noise_sound))  # нормуємо пік до 1
    L = len(noise_sound)
    assert L / sr >= 0.2, f"Шаблон має бути не коротшим за 200 мс (зараз {1000 * L / sr:.0f} мс)"

    # 2.2 Додаємо гаусівський шум
    sigma = np.sqrt(np.mean(noise_sound ** 2) / 10 ** (TEMPLATE_NOISE_SNR_DB / 10))
    template = noise_sound + rng.normal(0, sigma, L)
    print(f"Шаблон шуму: {1000 * L / sr:.0f} мс, σ гаусівського шуму = {sigma:.4f}")

    # ------------------------------------------------------------ 3. Додавання шаблону до сигналу
    # Позиції вставки в секундах -> у відліках
    positions = np.array(POSITIONS_SEC) * sr
    # Різні амплітуди на кожному входженні (відносно піку корисного сигналу)
    amplitudes = np.max(np.abs(x)) * rng.permutation(np.linspace(0.3, 1.0, N_OCCURRENCES))

    y = x.copy()
    for p, a in zip(positions, amplitudes):
        y[p:p + L] += a * template

    # ------------------------------------------------------------ 4. Пошук шаблону (нормована кореляція)
    corr = correlate(y, template, mode="valid")
    window_energy = np.convolve(y ** 2, np.ones(L), mode="valid")
    ncc = corr / (np.linalg.norm(template) * np.sqrt(window_energy) + 1e-12)
    # Адаптивний поріг: медіана + DETECT_Z * робастне σ (через MAD).
    # Клацання ручки розріджене й коротке, тому абсолютні значення NCC невеликі,
    # але піки на місцях шаблону в десятки σ вищі за фон.
    med = np.median(ncc)
    robust_sigma = 1.4826 * np.median(np.abs(ncc - med))
    threshold = med + DETECT_Z * robust_sigma
    found, _ = find_peaks(ncc, height=threshold, distance=L)
    print(f"Поріг NCC = {threshold:.4f}. Знайдено входжень: {len(found)} (фактично {N_OCCURRENCES})")

    # ------------------------------------------------------------ 5. Видалення шуму (МНК, псевдоінверсія)
    T = template.reshape(-1, 1)           # матриця L x 1
    T_pinv = np.linalg.pinv(T)            # псевдоінверсія Мура-Пенроуза, 1 x L
    cleaned = y.copy()
    est_amplitudes = []
    for p in found:
        a_hat = (T_pinv @ y[p:p + L]).item()   # a = (TᵀT)⁻¹ Tᵀ y
        est_amplitudes.append(a_hat)
        cleaned[p:p + L] -= a_hat * template

    # ------------------------------------------------------------ 6. Порівняння
    rows = []  # (№, справжня позиція, знайдена позиція | None, справжня a, оцінена a | None)
    for i, (p, a) in enumerate(zip(positions, amplitudes)):
        j = np.argmin(np.abs(found - p)) if len(found) else None
        if j is not None and abs(found[j] - p) < L:
            rows.append((i + 1, p / sr, found[j] / sr, a, est_amplitudes[j]))
        else:
            rows.append((i + 1, p / sr, None, a, None))

    print("\n  №   позиція (справжня / знайдена), с    амплітуда (справжня / оцінена)")
    for n, p_true, p_found, a_true, a_est in rows:
        if p_found is None:
            print(f"  {n}   {p_true:8.4f} /  не знайдено")
        else:
            print(f"  {n}   {p_true:8.4f} / {p_found:8.4f}               {a_true:.4f} / {a_est:.4f}")

    metrics = {
        "MSE": (np.mean((y - x) ** 2), np.mean((cleaned - x) ** 2)),
        "SNR, дБ": (snr_db(x, y), snr_db(x, cleaned)),
    }
    print(f"\nMSE  із шумом: {metrics['MSE'][0]:.3e}   після очищення: {metrics['MSE'][1]:.3e}")
    print(f"SNR  із шумом: {metrics['SNR, дБ'][0]:.2f} дБ   після очищення: {metrics['SNR, дБ'][1]:.2f} дБ")

    for name, sig in [("original.wav", x), ("noisy.wav", y), ("cleaned.wav", cleaned)]:
        wavfile.write(os.path.join(OUT_DIR, name), sr,
                      (np.clip(sig, -1, 1) * 32767).astype(np.int16))

    plot_results(sr, x, y, cleaned, noise_sound, template, ncc, threshold,
                 positions, found, L, rows, metrics)
    make_listen_page(positions / sr)


# ---------------------------------------------------------------- Графіки (plotly)
# Кольори: оригінал — синій, із шумом — помаранчевий, очищений — бірюзовий.
C_ORIG, C_NOISY, C_CLEAN, C_ERR = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


MAX_BUCKETS = 20000  # довші лінії проріджуємо: мільйон точок робить HTML на десятки МБ


def line(y, sr, name, color, width=1, dash=None, t0=0.0, **kw):
    """WebGL-лінія з рівномірною віссю часу (x0/dx замість масиву — легший HTML).

    Довгі сигнали проріджуються до пар (min, max) у кожному блоці відліків —
    форма хвилі та всі піки зберігаються, а розмір сторінки падає в десятки разів.
    """
    y = np.asarray(y, dtype=np.float32)
    style = dict(mode="lines", name=name, line=dict(color=color, width=width, dash=dash), **kw)
    if len(y) <= 5 * MAX_BUCKETS:  # коротким лініям проріджування не потрібне
        return go.Scattergl(y=y, x0=t0, dx=1 / sr, **style)
    step = -(-len(y) // MAX_BUCKETS)  # ділення з округленням угору
    n = len(y) // step * step
    blocks = y[:n].reshape(-1, step)
    idx = np.arange(0, n, step)
    lo, hi = blocks.argmin(axis=1) + idx, blocks.argmax(axis=1) + idx
    # Кожна пара — у часовому порядку, щоб лінія не йшла назад
    pts = np.sort(np.stack([lo, hi], axis=1), axis=1).ravel()
    if n < len(y):
        pts = np.append(pts, np.arange(n, len(y)))
    return go.Scattergl(x=(t0 + pts / sr).astype(np.float32), y=y[pts], **style)


def finish(fig, title, filename, height):
    fig.update_layout(title=title, height=height, template="plotly_white",
                      hovermode="closest", legend=dict(orientation="h", y=1.02, x=1,
                                                       xanchor="right", yanchor="bottom"))
    # Зберігаємо у файл і відкриваємо його в браузері.
    # (fig.show() піднімає одноразовий сервер, і частина вкладок іноді не завантажується.)
    path = os.path.abspath(os.path.join(OUT_DIR, filename))
    fig.write_html(path, include_plotlyjs="cdn")
    webbrowser.open(Path(path).as_uri())


def plot_results(sr, x, y, cleaned, noise_sound, template, ncc, threshold,
                 positions, found, L, rows, metrics):
    # Рис. 1 — шаблон шуму
    fig = go.Figure([
        line(template, sr, "шаблон + гаусівський шум", C_NOISY),
        line(noise_sound, sr, "звук клацання", C_ORIG),
    ])
    fig.update_xaxes(title="час, с")
    fig.update_yaxes(title="амплітуда")
    finish(fig, "Шаблон шуму", "1_template.html", 450)

    # Рис. 2 — сигнали (спільні осі: масштабування в одному вікні синхронізує всі)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=["Оригінальний корисний сигнал",
                                        "Сигнал із шумом (підсвічено — справжні місця вставки шаблону)",
                                        "Очищений сигнал",
                                        "Різниця: очищений − оригінальний"])
    fig.add_trace(line(x, sr, "оригінал", C_ORIG, 0.8), 1, 1)
    fig.add_trace(line(y, sr, "із шумом", C_NOISY, 0.8), 2, 1)
    fig.add_trace(line(cleaned, sr, "очищений", C_CLEAN, 0.8), 3, 1)
    fig.add_trace(line(cleaned - x, sr, "похибка", C_ERR, 0.8), 4, 1)
    for p in positions:
        fig.add_vrect(x0=p / sr, x1=(p + L) / sr, fillcolor=C_NOISY, opacity=0.15,
                      line_width=0, row=2, col=1)
    fig.update_xaxes(title="час, с", row=4, col=1)
    finish(fig, "Порівняння сигналів", "2_signals.html", 950)

    # Рис. 3 — нормована кореляція
    fig = go.Figure([
        line(ncc, sr, "NCC", C_ORIG, 0.8),
        go.Scatter(x=found / sr, y=ncc[found], mode="markers", name="знайдені піки",
                   marker=dict(symbol="triangle-down", size=12, color=C_NOISY,
                               line=dict(color="white", width=2)),
                   hovertemplate="зсув %{x:.4f} с<br>NCC %{y:.3f}<extra></extra>"),
    ])
    fig.add_hline(y=threshold, line_dash="dash", line_color=C_ERR,
                  annotation_text=f"поріг {threshold:.3f}", annotation_position="top left")
    fig.update_xaxes(title="зсув, с")
    fig.update_yaxes(title="NCC")
    finish(fig, "Нормована взаємна кореляція сигналу з шаблоном", "3_correlation.html", 500)

    # Рис. 4 — таблиця результатів
    fmt = lambda v, f: "—" if v is None else format(v, f)
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "table"}]],
                        row_heights=[0.65, 0.35], vertical_spacing=0.05)
    fig.add_trace(go.Table(
        header=dict(values=["№", "позиція справжня, с", "позиція знайдена, с",
                            "амплітуда справжня", "амплітуда оцінена"], align="center"),
        cells=dict(values=list(zip(*[(n, fmt(pt, ".4f"), fmt(pf, ".4f"), fmt(at, ".4f"), fmt(ae, ".4f"))
                                     for n, pt, pf, at, ae in rows])), align="center")), 1, 1)
    fig.add_trace(go.Table(
        header=dict(values=["метрика", "із шумом", "після очищення"], align="center"),
        cells=dict(values=[list(metrics),
                           [f"{v[0]:.3e}" if k == "MSE" else f"{v[0]:.2f}" for k, v in metrics.items()],
                           [f"{v[1]:.3e}" if k == "MSE" else f"{v[1]:.2f}" for k, v in metrics.items()]],
                   align="center")), 2, 1)
    finish(fig, "Результати", "4_results.html", 500)

    print(f"\nІнтерактивні графіки відкрито в браузері та збережено в папці '{OUT_DIR}/'")


# ---------------------------------------------------------------- Прослуховування
LISTEN_HTML = """<!doctype html>
<html lang="uk"><head><meta charset="utf-8"><title>Прослуховування</title>
<style>
  :root {{ --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#1a1a19; --fg:#fff; --muted:#c3c2b7; --line:#383835; --accent:#3987e5; }} }}
  body {{ background:var(--bg); color:var(--fg); font:16px/1.5 -apple-system, system-ui, sans-serif;
         max-width:760px; margin:40px auto; padding:0 16px; }}
  h1 {{ font-size:22px; }}  p {{ color:var(--muted); }}
  .track {{ border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin:12px 0; }}
  .track b {{ display:block; margin-bottom:6px; }}
  audio {{ width:100%; }}
  .jumps {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }}
  button {{ font:inherit; font-size:14px; padding:4px 10px; border-radius:6px; cursor:pointer;
           border:1px solid var(--line); background:transparent; color:var(--fg); }}
  button:hover {{ border-color:var(--accent); color:var(--accent); }}
</style></head><body>
<h1>Прослуховування: до та після очищення</h1>
<p>Кнопки під кожним плеєром перемотують до входження шаблону (за 0.5 с до нього) і запускають відтворення.
Одночасно грає лише одна доріжка.</p>
{tracks}
<script>
  const players = [...document.querySelectorAll("audio")];
  players.forEach(a => a.addEventListener("play", () => players.forEach(o => o !== a && o.pause())));
  document.querySelectorAll("button[data-t]").forEach(b => b.onclick = () => {{
    const a = document.getElementById(b.dataset.a);
    a.currentTime = Math.max(0, +b.dataset.t - 0.5); a.play();
  }});
</script></body></html>
"""


def make_listen_page(times):
    tracks = ""
    for file, label in [("original.wav", "Оригінал"), ("noisy.wav", "Із шумом (клацання ручки)"),
                        ("cleaned.wav", "Після очищення")]:
        aid = file.split(".")[0]
        jumps = "".join(f'<button data-a="{aid}" data-t="{t:.3f}">▶ {t:.2f} с</button>' for t in times)
        tracks += (f'<div class="track"><b>{label}</b><audio id="{aid}" controls preload="auto" '
                   f'src="{file}"></audio><div class="jumps">{jumps}</div></div>\n')
    path = os.path.abspath(os.path.join(OUT_DIR, "0_listen.html"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(LISTEN_HTML.format(tracks=tracks))
    webbrowser.open(Path(path).as_uri())


if __name__ == "__main__":
    main()
