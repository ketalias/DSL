# DSL — лабораторні з цифрової обробки сигналів

## Налаштування

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Лабораторні

| Лаба | Тема | Запуск |
|------|------|--------|
| `lab1/` | Видалення шуму за його шаблоном (кореляція) | `cd lab1 && python main.py` |
| `lab2/` | Розділення пташиного співу: STFT, FIR/IIR смугові фільтри | `cd lab2 && python main.py` |

Вхідні аудіо лежать у `labN/audio/`. Результати (`lab1/results/`, `lab2/output/`)
генеруються скриптами і в git не зберігаються.
