"""Запуск программы «Генератор ПИ».

Служебный режим для проверки сборки (без окна):
    GeneratorPI.exe --selftest отчёт.txt шаблон.docx фото1.jpg фото2.jpg ...
Распознаёт номера на фото и собирает рядом с отчётом файл «отчёт.docx»
(все компоненты — чистый полимер).
"""
import os
import sys


def selftest(report_path, template, images):
    from generator_pi.docx_builder import ComponentSpec, PhotoSpec, build
    from generator_pi.ocr import NumberReader, assign_ranges

    reader = NumberReader()
    numbers = [reader.read_numbers(p) for p in images]
    ranges = assign_ranges(numbers)
    with open(report_path, 'w', encoding='utf-8') as f:
        for path, ns, rng in zip(images, numbers, ranges):
            f.write(f'{os.path.basename(path)}\t{ns}\t{rng}\n')
        photos = sorted((PhotoSpec(p, r) for p, r in zip(images, ranges)),
                        key=lambda s: (s.range is not None, s.range[0] if s.range else 0))
        count = max((r[1] for r in ranges if r), default=0)
        out = os.path.splitext(report_path)[0] + '.docx'
        rep = build(template, out, 'Тестовый образец, марка "Тест", модель T-1', photos,
                    [ComponentSpec(n) for n in range(1, count + 1)], watermark=True)
        f.write(f'docx\t{out}\t{rep}\n')


if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[1] == '--selftest':
        selftest(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        from generator_pi.app import main
        main()
