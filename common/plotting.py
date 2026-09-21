from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_figure(fig, path, dpi: int = 200, extra_artists=None) -> None:
    """`extra_artists`: artists outside the normal axes bbox (e.g. a legend placed via
    ax.add_artist() at bbox_to_anchor=(1.02, 1)) that must also be included in the saved image.
    Without this, bbox_inches="tight" can silently crop such a legend out of the PNG entirely if
    the figure was already tight_layout()'d without knowing about it -- call fig.tight_layout()
    BEFORE creating any such extra legend, or not at all, when passing extra_artists here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)
