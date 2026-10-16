from .main import build


def view(height=900, **kw):
    # the page in a notebook; same arguments as build()
    import html
    from IPython.display import HTML
    page, _ = build(cdn=True, **kw)
    return HTML(f'<iframe srcdoc="{html.escape(page)}" style="width:100%;height:{height}px;border:0"></iframe>')
