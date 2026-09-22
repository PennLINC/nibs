"""Publication-facing labels for regional parcel and bundle identifiers."""

from __future__ import annotations

import re

from make_autotrack_bundle_html_table import parse_track_id


def wm_bundle_display_label(track_id: object) -> str:
    """Convert an AutoTrack identifier to an anatomical name and laterality."""

    parsed = parse_track_id(str(track_id))
    label = parsed.bundle.title()
    if parsed.crossed_direction == 'lr':
        return f'{label} (L→R)'
    if parsed.crossed_direction == 'rl':
        return f'{label} (R→L)'
    if parsed.side in {'L', 'R'}:
        return f'{label} ({parsed.side})'
    return label


def dkt_parcel_display_label(parcel_id: object) -> str:
    """Convert a DKT ``lh_/rh_`` identifier to a readable cortical label."""

    value = str(parcel_id).strip()
    match = re.match(r'^(lh|rh|l|r)[_\s-]+(.+)$', value, flags=re.IGNORECASE)
    if match is None:
        return value.replace('_', ' ').title()

    hemisphere = 'L' if match.group(1).lower() in {'lh', 'l'} else 'R'
    region = re.sub(r'[_\s-]+', ' ', match.group(2)).strip().title()
    if not region.lower().endswith(' cortex'):
        region = f'{region} Cortex'
    return f'{region} ({hemisphere})'


def regional_feature_display_label(feature: object, tissue: str) -> str:
    """Return the appropriate publication label for a regional feature."""

    if tissue == 'wm':
        return wm_bundle_display_label(feature)
    if tissue == 'gm':
        return dkt_parcel_display_label(feature)
    raise ValueError(f'Unsupported regional tissue: {tissue}')
