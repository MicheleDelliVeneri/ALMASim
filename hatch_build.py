"""Put the built web app into the distributions (web console plan, S2.6).

``npm run build`` in ``frontend/`` writes the app to ``frontend/build``. The
sdist carries that folder as it is, and the wheel carries it as
``almasim/web/static``, where ``almasim serve`` finds it. Neither runs npm:
build the app first (CI and the release workflow do). Without a build the
distributions are made without the app, and the console then says its pages
are not built, unless ``ALMASIM_REQUIRE_WEB=1``, which makes that an error
(the release sets it, so no wheel ships without its app).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

BUILD = Path("frontend") / "build"
PAGE = "200.html"  # adapter-static's fallback page: the app is built
REQUIRE_ENV = "ALMASIM_REQUIRE_WEB"


class WebAppHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name == "wheel" and version == "editable":
            return  # a source checkout serves frontend/build directly
        build = Path(self.root) / BUILD
        if not (build / PAGE).is_file():
            if os.environ.get(REQUIRE_ENV) == "1":
                raise RuntimeError(
                    f"{BUILD} has no built web app; run `npm ci && npm run build` in frontend/"
                )
            self.app.display_warning(f"{BUILD} is not built: the console will have no pages")
            return
        target = "almasim/web/static" if self.target_name == "wheel" else str(BUILD)
        build_data["force_include"][str(build)] = target
