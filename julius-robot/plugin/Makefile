CC       = gcc
JDIR     = ./julius-4.3.1
IDIR1    = $(JDIR)/libjulius/include
IDIR2    = $(JDIR)/libsent/include
IDIR3    = ./zlib-1.2.3-lib/include
LDIR1    = $(JDIR)/libjulius
LDIR2    = $(JDIR)/libsent
LDIR3    = ./zlib-1.2.3-lib/lib
LIBS     = `$(JDIR)/libjulius/libjulius-config --libs` `$(JDIR)/libsent/libsent-config --libs` 
DEFINES  = `$(JDIR)/libjulius/libjulius-config --cflags` `$(JDIR)/libsent/libsent-config --cflags` 
PROGNAME = okaeri_kanojo

all : $(PROGNAME).jpi

$(PROGNAME).jpi : $(PROGNAME).o
	$(CC) -shared -o $(PROGNAME).jpi $(PROGNAME).o -L$(LDIR1) -L$(LDIR2) -L$(LDIR3) $(LIBS)

$(PROGNAME).o : $(PROGNAME).c
	$(CC) -c $(PROGNAME).c -Wall -I$(IDIR1) -I$(IDIR2) -I$(IDIR3) $(DEFINES)

.PHONY: clean
clean :
	-rm $(PROGNAME).jpi $(PROGNAME).o

