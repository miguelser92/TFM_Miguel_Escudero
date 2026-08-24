# =====================================================================
#  Configuracion de latexmk para esta carpeta
# =====================================================================
#  Cualquier herramienta que compile con latexmk lee esto automaticamente:
#  el script compilar.ps1, la extension de VS Code, TeXstudio, TeXworks...
#  Asi los auxiliares van a .build\ aunque no se use el script.
#
#  Los .aux, .bbl, .out y compania no son basura: son las notas que el
#  compilador se deja entre pasadas para resolver las referencias cruzadas
#  y la bibliografia. Pero no tienen por que estar a la vista.
# =====================================================================

# Auxiliares a .build, PDF en la carpeta (que es donde lo busca el visor).
$aux_dir = '.build';
$out_dir = '.';

# pdflatex + bibtex, sin parar en el primer aviso.
$pdf_mode = 1;
$bibtex_use = 2;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# 'latexmk -c' limpia tambien estas extensiones.
$clean_ext = 'bbl nav out snm synctex.gz fdb_latexmk fls run.xml';
