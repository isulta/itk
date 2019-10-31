# command line utility to inspect GenericIO file
import genericio
import sys

#print(sys.argv)

if len(sys.argv) < 2:
    print("No genericio file specified.")
else:
    fname = sys.argv[1]
    genericio.gio_inspect(fname)


