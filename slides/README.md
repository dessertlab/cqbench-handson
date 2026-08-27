# Opening deck

`index.html` — the standalone deck. Open it in any browser; it needs no server and
no build step. Fonts load from Google Fonts when online and fall back gracefully
when not, so it works in a conference room with no wifi.

**Controls:** `→` `←` / `space` / PageUp–PageDown (works with a presenter clicker) ·
`Home` / `End` · `f` for fullscreen · swipe on touch. The URL hash tracks the slide,
so you can link straight to one. `Cmd/Ctrl-P` prints one slide per page.

It follows the viewer's light/dark preference.

`_body.html` is the same deck without the outer `<html>/<head>/<body>` wrapper —
that's the version published as a hosted page. Edit `_body.html` and regenerate
`index.html` if you change the deck.
