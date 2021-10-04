# This script is copyed from https://stackoverflow.com/questions/55057888/difficult-workflow-writing-latex-book-full-of-python-code
# Auhtor: ymonad (Stack Overflow)
import sys
import re
import os
from collections import defaultdict
import nbformat
from nbconvert import LatexExporter, exporters

OUTPUT_FILES_DIR = './images'
CUSTOM_TEMPLATE = 'style_ipython_custom.tplx'
MAIN_TEX = 'main.tex'


def create_main():
    # creates `main.tex` which only has macro definition
    latex_exporter = LatexExporter()
    book = nbformat.v4.new_notebook()
    book.cells.append(
        nbformat.v4.new_raw_cell(r'\input{__your_input__here.tex}'))
    (body, _) = latex_exporter.from_notebook_node(book)
    with open(MAIN_TEX, 'x') as fout:
        fout.write(body)
    print("created:", MAIN_TEX)


def init():
    create_main()
    latex_exporter = LatexExporter()
    # copy `style_ipython.tplx` in `nbconvert.exporters` module to current directory,
    # and modify it so that it does not contain macro definition
    tmpl_path = os.path.join(
        os.path.dirname(exporters.__file__),
        latex_exporter.default_template_path)
    src = os.path.join(tmpl_path, 'style_ipython.tplx')
    target = CUSTOM_TEMPLATE
    with open(src) as fsrc:
        with open(target, 'w') as ftarget:
            for line in fsrc:
                # replace the line so than it does not contain macro definition
                if line == "((*- extends 'base.tplx' -*))\n":
                    line = "((*- extends 'document_contents.tplx' -*))\n"
                ftarget.write(line)
    print("created:", CUSTOM_TEMPLATE)


def group_cells(note):
    # scan the cell source for tag with regexp `^#latex:(.*)`
    # if sames tags are found group it to same list
    pattern = re.compile(r'^#latex:(.*?)$(\n?)', re.M)
    group = defaultdict(list)
    for num, cell in enumerate(note.cells):
        m = pattern.search(cell.source)
        if m:
            tag = m.group(1).strip()
            # remove the line which contains tag
            cell.source = cell.source[:m.start(0)] + cell.source[m.end(0):]
            group[tag].append(cell)
        else:
            print("tag not found in cell number {}. ignore".format(num + 1))
    return group


def doit():
    with open(sys.argv[1]) as f:
        note = nbformat.read(f, as_version=4)
    group = group_cells(note)
    latex_exporter = LatexExporter()
    # use the template which does not contain LaTex macro definition
    latex_exporter.template_file = CUSTOM_TEMPLATE
    try:
        os.mkdir(OUTPUT_FILES_DIR)
    except FileExistsError:
        pass
    for (tag, g) in group.items():
        book = nbformat.v4.new_notebook()
        book.cells.extend(g)
        # unique_key will be prefix of image
        (body, resources) = latex_exporter.from_notebook_node(
            book,
            resources={
                'output_files_dir': OUTPUT_FILES_DIR,
                'unique_key': tag
            })
        ofile = tag + '.tex'
        with open(ofile, 'w') as fout:
            fout.write(body)
            print("created:", ofile)
        # the image data which is embedded as base64 in notebook
        # will be decoded and returned in `resources`, so write it to file
        for filename, data in resources.get('outputs', {}).items():
            with open(filename, 'wb') as fres:
                fres.write(data)
                print("created:", filename)


if len(sys.argv) <= 1:
    print("USAGE: this_script [init|yourfile.ipynb]")
elif sys.argv[1] == "init":
    init()
else:
    doit()