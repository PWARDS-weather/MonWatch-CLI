# =============================================================================
# rem_ingest — MonWatch-CLI remote sensing ingest package
#
# (C) 2025-2026 PWARDS-weather
# SPDX-License-Identifier: Apache-2.0 OR GPL-3.0-or-later
#
# Dual-licensed. You may use, modify, and distribute this package under the
# terms of EITHER the Apache License, Version 2.0, or the GNU General Public
# License, Version 3.0 or later — not both. See LICENSE.txt in the project
# root for the full license texts and the copyright notice.
# =============================================================================
from . import common
from . import _rgb_corrections
from . import himawari
from . import gk2a
from . import goes
from . import mtg
from . import mtsat
from . import jpss_common
from . import jpss_class
from . import jpss_pds
from . import jpss


__all__ = [
    "common", "_rgb_corrections",
    "himawari", "gk2a", "goes", "mtg", "mtsat",
    "jpss_common", "jpss_class", "jpss_pds", "jpss",
]