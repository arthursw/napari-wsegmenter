from napari import Viewer, run


def main() -> None:
    """Launch napari and let its plugin menu manage dock widgets."""
    Viewer()
    run()


if __name__ == "__main__":
    main()
