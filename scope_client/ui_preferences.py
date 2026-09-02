from __future__ import annotations

import ctypes
import sys
from typing import Dict

from PyQt5 import QtCore, QtGui, QtWidgets

from .theme import APP_QSS


# Keep translation deliberately UI-only: protocol names, wheel names, units and
# saved project data are never rewritten.
_EN_PT: Dict[str, str] = {
    "&File": "&Arquivo", "&Tools": "&Ferramentas", "&View": "&Exibir", "&Appearance": "&Aparência",
    "Theme": "Tema", "Language": "Idioma", "Dark": "Escuro", "Legacy PyQt5": "PyQt5 legado",
    "About OpenScope…": "Sobre o OpenScope…", "About OpenScope": "Sobre o OpenScope",
    "Portuguese": "Português", "English": "Inglês",
    "Open capture (.npz)…": "Abrir captura (.npz)…", "Save capture (.npz)…": "Salvar captura (.npz)…",
    "Export CSV…": "Exportar CSV…", "Export plot PNG…": "Exportar gráfico PNG…",
    "Save settings…": "Salvar configurações…", "Load settings…": "Carregar configurações…", "Exit": "Sair",
    "Visit Lobby": "Ir para o lobby", "FFT / Spectrum…": "FFT / Espectro…", "XY Mode...": "Modo XY...",
    "RPM / Frequency Calculator…": "Calculadora RPM / frequência…",
    "Electrical Network Analyzer…": "Analisador de rede elétrica…", "Electrical Network Analyzer": "Analisador de rede elétrica",
    "Phase Sequence…": "Sequência de fase…", "Phase Sequence": "Sequência de fase",
    "&Plugins": "&Plugins", "Plugin manager…": "Gerenciador de plugins…", "Open plugins folder": "Abrir pasta de plugins",
    "OpenScope Plugins": "Plugins do OpenScope", "Open folder": "Abrir pasta", "Load selected": "Carregar selecionado",
    "Phase reference": "Referência de fase", "Three channels are required": "São necessários três canais",
    "Select three different channels": "Selecione três canais diferentes", "Sequence unavailable": "Sequência indisponível",
    "Auto Set": "Ajuste automático", "Reset active channel calibration": "Resetar calibração do canal ativo",
    "Reset vertical view": "Resetar visualização vertical", "Soft touch graph": "Interação suave no gráfico",
    "Δ Time cursors": "Cursores Δ tempo", "Δ Amplitude cursors": "Cursores Δ amplitude",
    "Compact HD layout": "Layout HD compacto", "Reset layout": "Resetar layout", "Full screen": "Tela cheia",
    "Channels": "Canais", "Graph": "Gráfico", "Graph actions": "Ações do gráfico", "Interaction": "Interação",
    "View X": "Visual X", "View Y": "Visual Y", "Grid height": "Altura da grade", "Connection": "Conexão",
    "Refresh": "Atualizar", "Connect": "Conectar", "Disconnect": "Desconectar", "Device state": "Estado do dispositivo",
    "Not connected": "Não conectado", "Port": "Porta", "Wheel": "Roda", "Wheel pattern": "Padrão da roda",
    "Pattern": "Padrão", "View": "Visualização", "Example": "Exemplo", "Mode": "Modo",
    "Potentiometer": "Potenciômetro", "Fixed RPM": "RPM fixo", "Ranged sweep": "Varredura de RPM",
    "Sweep min": "RPM mínimo", "Sweep max": "RPM máximo", "Sweep interval": "Intervalo da varredura",
    "Compression waves": "Ondas de compressão", "Enable compression waves": "Ativar ondas de compressão",
    "Dynamic compression": "Compressão dinâmica", "RPM threshold": "Limite de RPM", "Offset": "Offset",
    "Trigger": "Trigger", "Acquisition": "Aquisição", "Timebase": "Base de tempo", "Record/history": "Registro/histórico",
    "Persistence": "Persistência", "Sample rate": "Taxa de amostragem", "Source": "Fonte", "Edge": "Borda",
    "Level": "Nível", "Pre-trigger": "Pré-trigger", "Auto trigger": "Trigger automático", "Holdoff": "Holdoff",
    "Auto timeout": "Timeout automático", "Cursors": "Cursores", "Amplitude CH": "Amplitude CH",
    "Δ time": "Δ tempo", "Δ amplitude": "Δ amplitude", "ARM": "ARMAR", "TRIG": "TRIG", "ON": "LIGADO",
    "Measurements": "Medições", "Channel": "Canal", "Frequency": "Frequência", "Magnitude": "Magnitude",
    "RUN": "RODAR", "PAUSE": "PAUSAR", "STOP": "PARAR", "Demo": "Demo",
    "Select an Ardu-Stim port and connect.": "Selecione uma porta Ardu-Stim e conecte.",
    "Ports refreshed. Choose the Ardu-Stim port.": "Portas atualizadas. Escolha a porta do Ardu-Stim.",
    "No serial ports found for Ardu-Stim.": "Nenhuma porta serial encontrada para o Ardu-Stim.",
    "Select a port before connecting.": "Selecione uma porta antes de conectar.",
    "Ardu-Stim connected and configuration verified.": "Ardu-Stim conectado e configuração validada.",
    "Ardu-Stim disconnected.": "Ardu-Stim desconectado.", "No wheel pattern loaded": "Nenhum padrão de roda carregado",
    "OpenScope — Professional Oscilloscope": "OpenScope — Osciloscópio profissional",
    "OpenScope — Professional Oscilloscope · BY Valdemir": "OpenScope — Osciloscópio profissional · BY Valdemir",
    "OpenScope Lobby": "Lobby do OpenScope",
    "Choose a device": "Escolha um dispositivo", "Devices": "Dispositivos", "Open": "Abrir", "Cancel": "Cancelar",
    "Open through serial or enter Demo mode to test the main screen.": "Abra pela serial ou entre em modo Demo para testar a tela principal.",
    "&Acquisition": "&Aquisição",
    "&Help": "A&juda",
    "Run acquisition": "Iniciar aquisição",
    "Pause acquisition": "Pausar aquisição",
    "Stop acquisition": "Parar aquisição",
    "Re-arm trigger": "Rearmar trigger",
    "Force trigger": "Forçar trigger",
    "Refresh serial ports": "Atualizar portas seriais",
    "Connect / Disconnect serial": "Conectar / desconectar serial",
}
_PT_EN = {v: k for k, v in _EN_PT.items()}


def tr(text: str, language: str) -> str:
    if language == "pt":
        return _EN_PT.get(text, text)
    return _PT_EN.get(text, text)


def _key(obj, value: str) -> str:
    stored = obj.property("_scp_i18n_key")
    if stored is None:
        # Normalize an already-Portuguese initial string back to the canonical
        # English key where one is known.
        stored = _PT_EN.get(value, value)
        obj.setProperty("_scp_i18n_key", stored)
    return str(stored)


def translate_ui(root: QtWidgets.QWidget, language: str) -> None:
    for widget in [root] + root.findChildren(QtWidgets.QWidget):
        if isinstance(widget, (QtWidgets.QLabel, QtWidgets.QPushButton, QtWidgets.QToolButton,
                               QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
            value = widget.text()
            # Dynamic labels (device values/statuses) are left alone unless the
            # entire current text is a known static phrase.
            canonical = _PT_EN.get(value, value)
            if canonical in _EN_PT or value in _EN_PT:
                widget.setText(tr(_key(widget, value), language))
        elif isinstance(widget, QtWidgets.QGroupBox):
            widget.setTitle(tr(_key(widget, widget.title()), language))
        elif isinstance(widget, QtWidgets.QDockWidget):
            widget.setWindowTitle(tr(_key(widget, widget.windowTitle()), language))
        elif isinstance(widget, QtWidgets.QTabWidget):
            for i in range(widget.count()):
                page = widget.widget(i)
                prop = f"_scp_tab_key_{i}"
                canonical = page.property(prop)
                if canonical is None:
                    txt = widget.tabText(i)
                    canonical = _PT_EN.get(txt, txt)
                    page.setProperty(prop, canonical)
                widget.setTabText(i, tr(str(canonical), language))
        elif isinstance(widget, QtWidgets.QComboBox):
            for i in range(widget.count()):
                txt = widget.itemText(i)
                canonical = _PT_EN.get(txt, txt)
                if canonical in _EN_PT:
                    widget.setItemText(i, tr(canonical, language))

    for action in root.findChildren(QtWidgets.QAction):
        text = action.text()
        canonical = _PT_EN.get(text, text)
        if canonical in _EN_PT or text in _EN_PT:
            action.setText(tr(_key(action, text), language))
    for menu in root.findChildren(QtWidgets.QMenu):
        menu.setTitle(tr(_key(menu, menu.title()), language))

    title = root.windowTitle()
    canonical = _PT_EN.get(title, title)
    if canonical in _EN_PT:
        root.setWindowTitle(tr(_key(root, title), language))


def set_windows_dark_titlebar(widget: QtWidgets.QWidget, enabled: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        value = ctypes.c_int(1 if enabled else 0)
        dwm = ctypes.windll.dwmapi
        # 20 is current Windows 10/11; 19 is used by older Windows 10 builds.
        result = dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        if result != 0:
            dwm.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass


def _dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#10161d"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e6edf6"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#0d131a"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#121a24"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#111923"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#e6edf6"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#e6edf6"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#1a2433"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#e6edf6"))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#315b86"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor("#67b6ff"))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor("#6f8094"))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor("#6f8094"))
    return palette


def apply_theme(app: QtWidgets.QApplication, theme: str) -> str:
    theme = "legacy" if theme == "legacy" else "dark"
    if theme == "dark":
        app.setStyle("Fusion")
        app.setPalette(_dark_palette())
        app.setStyleSheet(APP_QSS)
    else:
        app.setStyleSheet("")
        keys = {k.lower(): k for k in QtWidgets.QStyleFactory.keys()}
        style = keys.get("windowsvista") or keys.get("windows") or keys.get("fusion")
        if style:
            app.setStyle(style)
        app.setPalette(app.style().standardPalette())
    for widget in app.topLevelWidgets():
        if widget.isWindow():
            set_windows_dark_titlebar(widget, theme == "dark")
            widget.update()
    return theme
