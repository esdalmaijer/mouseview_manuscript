import os

DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data", "gaze", "split_files")
OUTDIR = os.path.join(DIR, "data", "gaze")

all_files = os.listdir(DATADIR)
ppnames = []
for fname in all_files:
    name, ext = os.path.splitext(fname)
    if ext != ".tsv":
        continue
    ppname = name[:11]
    if ppname not in ppnames:
        ppnames.append(ppname)

ppnames.sort()
for i, ppname in enumerate(ppnames):
    this_pp_files = []
    for fname in all_files:
        if ppname in fname:
            this_pp_files.append(fname)
    if len(this_pp_files) > 1:
        this_pp_files.sort()
        with open(os.path.join(DATADIR, this_pp_files[0]), "r") as f:
            content = f.readlines()
        for j in range(1, len(this_pp_files)):
            with open(os.path.join(DATADIR, this_pp_files[j]), "r") as f:
                lines = f.readlines()
                lines.pop(0)
                content.extend(lines)

        print("Writing combined file for '{}'".format(ppname))
        with open(os.path.join(OUTDIR, ppname+".tsv"), "w") as f:
            for line in content:
                f.write(line)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                
