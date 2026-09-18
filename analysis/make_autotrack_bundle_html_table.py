#!/usr/bin/env python3
"""Create a categorized, bilateral-aware HTML table of DSI Studio AutoTrack bundles."""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path


CATEGORY_LABELS = {
    'Association': 'Association',
    'ProjectionBasalGanglia': 'Projection—Basal Ganglia',
    'ProjectionBrainstem': 'Projection—Brainstem',
    'Commissure': 'Commissural',
    'Cerebellum': 'Cerebellar',
}

CATEGORY_COLORS = {
    'Association': '#3F6FA8',
    'ProjectionBasalGanglia': '#B13F82',
    'ProjectionBrainstem': '#268A4B',
    'Commissure': '#E86F2A',
    'Cerebellum': '#7B4D9E',
}


@dataclass(frozen=True)
class ParsedTrack:
    category: str
    bundle_key: str
    bundle: str
    side: str | None
    crossed_direction: str | None
    track_id: str


@dataclass(frozen=True)
class BundleRow:
    category: str
    bundle: str
    laterality: str
    track_ids: tuple[str, ...]


def extract_track_ids(spec_file: Path) -> list[str]:
    """Read the AutoTrack ``track_id`` list without requiring PyYAML."""

    lines = spec_file.read_text(encoding='utf-8').splitlines()
    candidates = [line for line in lines if line.lstrip().startswith('track_id:')]
    if len(candidates) != 1:
        raise RuntimeError(
            f'Expected exactly one track_id entry in {spec_file}; found {len(candidates)}.'
        )
    value = candidates[0].split(':', 1)[1].strip()
    track_ids = [token.strip() for token in value.split(',') if token.strip()]
    if not track_ids:
        raise RuntimeError(f'No AutoTrack bundle identifiers found in {spec_file}.')
    return track_ids


def humanize_bundle(value: str) -> str:
    words = value.replace('_', ' ')
    words = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', words)
    words = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', words)
    words = re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', words)
    words = re.sub(r'\s+', ' ', words).strip().lower()
    replacements = {
        'fronto occipital': 'fronto-occipital',
        'parahippocampal parietal': 'parahippocampal–parietal',
        'dentatorubrothalamic': 'dentato-rubro-thalamic',
        'non decussating': 'non-decussating',
        'parolfactory': 'parolfactory',
    }
    for source, target in replacements.items():
        words = words.replace(source, target)
    return words[:1].upper() + words[1:]


def parse_track_id(track_id: str) -> ParsedTrack:
    if '_' not in track_id:
        raise ValueError(f'Unexpected AutoTrack identifier without category: {track_id}')
    category, raw_bundle = track_id.split('_', 1)

    crossed_direction = None
    crossed_match = re.search(r'-(lr|rl)$', raw_bundle, flags=re.IGNORECASE)
    if crossed_match:
        crossed_direction = crossed_match.group(1).lower()
        raw_bundle = raw_bundle[: crossed_match.start()]

    side_match = re.search(r'(?<=[a-z0-9])([LR])(?=_|$)', raw_bundle)
    side = side_match.group(1) if side_match else None
    if side_match:
        raw_bundle = raw_bundle[: side_match.start()] + raw_bundle[side_match.end():]

    bundle_key = re.sub(r'[^A-Za-z0-9]+', '_', raw_bundle).strip('_')
    return ParsedTrack(
        category=category,
        bundle_key=bundle_key,
        bundle=humanize_bundle(raw_bundle),
        side=side,
        crossed_direction=crossed_direction,
        track_id=track_id,
    )


def collapse_bilateral_tracks(track_ids: list[str]) -> list[BundleRow]:
    grouped: dict[tuple[str, str], list[ParsedTrack]] = {}
    category_order: list[str] = []
    for track_id in track_ids:
        parsed = parse_track_id(track_id)
        if parsed.category not in category_order:
            category_order.append(parsed.category)
        grouped.setdefault((parsed.category, parsed.bundle_key), []).append(parsed)

    rows: list[BundleRow] = []
    for category in category_order:
        category_groups = [
            tracks for (group_category, _), tracks in grouped.items()
            if group_category == category
        ]
        for tracks in category_groups:
            sides = {track.side for track in tracks if track.side is not None}
            directions = {
                track.crossed_direction
                for track in tracks
                if track.crossed_direction is not None
            }
            if directions == {'lr', 'rl'}:
                laterality = 'Crossed bilateral (L→R and R→L)'
            elif directions:
                direction = next(iter(directions))
                laterality = 'Crossed (L→R)' if direction == 'lr' else 'Crossed (R→L)'
            elif sides == {'L', 'R'}:
                laterality = 'Bilateral'
            elif sides == {'L'}:
                laterality = 'Left'
            elif sides == {'R'}:
                laterality = 'Right'
            else:
                laterality = 'Midline / unpaired'
            rows.append(
                BundleRow(
                    category=category,
                    bundle=tracks[0].bundle,
                    laterality=laterality,
                    track_ids=tuple(track.track_id for track in tracks),
                )
            )
    return rows


def tinted_background(color: str, alpha: float = 0.10) -> str:
    token = color.lstrip('#')
    rgb = tuple(int(token[index:index + 2], 16) for index in range(0, 6, 2))
    tint = tuple(round(255 * (1 - alpha) + channel * alpha) for channel in rgb)
    return f'rgb({tint[0]}, {tint[1]}, {tint[2]})'


def write_tsv(rows: list[BundleRow], output: Path) -> None:
    lines = ['category\tbundle\tlaterality\tautotrack_ids']
    for row in rows:
        lines.append(
            '\t'.join(
                [
                    CATEGORY_LABELS.get(row.category, row.category),
                    row.bundle,
                    row.laterality,
                    ','.join(row.track_ids),
                ]
            )
        )
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_html(rows: list[BundleRow], output: Path, spec_file: Path) -> None:
    categories = list(dict.fromkeys(row.category for row in rows))
    body: list[str] = []
    for category in categories:
        category_rows = [row for row in rows if row.category == category]
        color = CATEGORY_COLORS.get(category, '#777777')
        tint = tinted_background(color)
        label = CATEGORY_LABELS.get(category, category)
        body.append(
            '<tr class="category-row" '
            f'style="background:{tint};border-left:0.42rem solid {color}">'
            f'<th colspan="2">{html.escape(label)}'
            f'<span>{len(category_rows)} bundles</span></th></tr>'
        )
        for row in category_rows:
            ids = ', '.join(row.track_ids)
            body.append(
                '<tr class="bundle-row" '
                f'style="background:{tint};border-left:0.42rem solid {color}" '
                f'title="{html.escape(ids)}">'
                f'<td>{html.escape(row.bundle)}</td>'
                f'<td>{html.escape(row.laterality)}</td></tr>'
            )

    document = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSI Studio AutoTrack bundles</title>
<style>
  :root {{ font-family: Arial, Helvetica, sans-serif; color: #202124; }}
  body {{ margin: 0; background: #f4f5f7; }}
  main {{ max-width: 980px; margin: 2rem auto; padding: 2rem 2.2rem; background: white;
          border-radius: 14px; box-shadow: 0 10px 30px rgba(28,39,54,.10); }}
  h1 {{ margin: 0 0 .35rem; font-size: 1.75rem; }}
  .subtitle {{ margin: 0 0 1.3rem; color: #5f6368; line-height: 1.45; }}
  table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .98rem; }}
  thead th {{ position: sticky; top: 0; z-index: 2; background: #263238; color: white;
              text-align: left; padding: .72rem .85rem; letter-spacing: .02em; }}
  thead th:first-child {{ width: 72%; }}
  td {{ padding: .56rem .85rem; border-bottom: 1px solid rgba(0,0,0,.08); }}
  .category-row th {{ padding: .64rem .78rem; text-align: left; font-size: 1.02rem; }}
  .category-row span {{ float: right; color: #5f6368; font-size: .86rem; font-weight: normal; }}
  .bundle-row:hover {{ filter: brightness(.97); }}
  .note {{ margin-top: 1rem; font-size: .84rem; color: #5f6368; line-height: 1.45; }}
  @media print {{
    body {{ background: white; }} main {{ max-width: none; margin: 0; padding: 0; box-shadow: none; }}
    thead {{ display: table-header-group; }} .category-row {{ break-after: avoid; }}
  }}
</style>
</head>
<body><main>
<h1>DSI Studio AutoTrack bundles</h1>
<p class="subtitle">Bilateral homologues are collapsed into one row. The table represents
{len(extract_track_ids(spec_file))} AutoTrack identifiers as {len(rows)} anatomical bundle rows,
organized by tract category.</p>
<table>
<thead><tr><th>Bundle</th><th>Laterality</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<p class="note">Source: <code>{html.escape(str(spec_file))}</code>. Hover over a row to view the
exact DSI Studio AutoTrack identifier(s). “Midline / unpaired” denotes bundles without a left/right
identifier in the reconstruction specification.</p>
</main></body></html>'''
    output.write_text(document, encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--spec-file',
        type=Path,
        default=repo_root / 'processing' / 'qsirecon_spec.yml',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=repo_root / 'figures' / 'supplemental_autotrack_bundle_table.html',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec_file = args.spec_file.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = collapse_bilateral_tracks(extract_track_ids(spec_file))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_html(rows, output, spec_file)
    tsv_output = output.with_suffix('.tsv')
    write_tsv(rows, tsv_output)
    print(f'Wrote: {output}')
    print(f'Wrote: {tsv_output}')


if __name__ == '__main__':
    main()
