# Bundled fonts

The app loads no fonts from the network.

- `manrope-latin.woff2`, `jetbrains-mono-latin.woff2`: Latin subsets from Google Fonts (OFL).
- `material-symbols-outlined.woff2`: a subset of Material Symbols Outlined (Apache 2.0) that
  contains only the icons the app uses.

## Adding an icon

An icon missing from the subset renders as its raw name (for example `space_dashboard`). To add
one, regenerate the subset with every icon name used in `src/` plus the new one. Request it from
the Google Fonts CSS API (sorted names, comma separated) and save the `woff2` it points to:

```
https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&icon_names=add,arrow_forward,...&display=block
```

`npm test` checks that every icon name used in the source is in `icon-names.json`, so the list
and the font cannot drift apart silently.
