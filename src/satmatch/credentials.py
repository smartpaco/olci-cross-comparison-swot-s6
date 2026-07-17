from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiCredentials:
    eumetsat_key: str
    eumetsat_secret: str
    earthdata_token: str


def load_api_credentials(path: str | Path) -> ApiCredentials:
    """Charge le fichier local sans jamais journaliser les secrets."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines()]

    def after_label(fragment: str) -> str:
        for index, line in enumerate(lines):
            if fragment in line.casefold():
                inline = line.split(":", 1)[1].strip() if ":" in line else ""
                if inline:
                    return inline
                for candidate in lines[index + 1 :]:
                    if candidate:
                        return candidate
        raise ValueError(f"Champ {fragment!r} absent du fichier de clés")

    key = after_label("key")
    secret = after_label("secret")
    token = after_label("nasa")
    if len(key) < 10 or len(secret) < 10 or len(token) < 40:
        raise ValueError("Le fichier de clés contient une valeur manifestement incomplète")
    return ApiCredentials(key, secret, token)

