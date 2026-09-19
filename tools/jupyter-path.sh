# JupyterLab venv (/pok8/jupyter)
if [ -d /pok8/jupyter/.venv/bin ]; then
    case ":$PATH:" in
        *":/pok8/jupyter/.venv/bin:"*) ;;
        *) PATH="/pok8/jupyter/.venv/bin:$PATH" ;;
    esac
    export PATH
fi
