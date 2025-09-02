def set_chinese_font():
    # 设置matplotlib支持中文显示
    import matplotlib
    import matplotlib.font_manager as fm
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'SC' in f.name]
    print("可用中文字体:", chinese_fonts)

    if 'SimHei' not in chinese_fonts:
        chinese_fonts += ['SimHei']

    matplotlib.rcParams['font.sans-serif'] = chinese_fonts # 设置中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    return chinese_fonts