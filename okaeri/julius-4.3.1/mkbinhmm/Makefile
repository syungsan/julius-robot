# Generated automatically from Makefile.in by configure.
# Copyright (c) 2003-2013 Kawahara Lab., Kyoto University 
# Copyright (c) 2003-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2013 Julius project team, Nagoya Institute of Technology
# All rights reserved
#
# $Id: Makefile.in,v 1.9 2013/06/20 17:14:27 sumomo Exp $
#
SHELL=/bin/sh
.SUFFIXES:
.SUFFIXES: .c .o
.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ -c $<

LIBSENT=../libsent
CC=gcc
CFLAGS=-O6 -fomit-frame-pointer
CPPFLAGS=-I$(LIBSENT)/include    `$(LIBSENT)/libsent-config --cflags`
LDFLAGS= -L$(LIBSENT) `$(LIBSENT)/libsent-config --libs`
RM=/bin/rm -f
prefix=/home/tadan/julius
exec_prefix=${prefix}
INSTALL=/bin/install -c

all: mkbinhmm.exe mkbinhmmlist.exe

mkbinhmm.exe: mkbinhmm.o $(LIBSENT)/libsent.a
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ mkbinhmm.o $(LDFLAGS)

mkbinhmmlist.exe: mkbinhmmlist.o $(LIBSENT)/libsent.a
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ mkbinhmmlist.o $(LDFLAGS)

install: install.bin

install.bin: mkbinhmm.exe mkbinhmmlist.exe
	${INSTALL} -d ${exec_prefix}/bin
	${INSTALL} mkbinhmm.exe mkbinhmmlist.exe ${exec_prefix}/bin

clean:
	$(RM) mkbinhmm.o mkbinhmmlist.o
	$(RM) *~ core
	$(RM) mkbinhmm mkbinhmm.exe
	$(RM) mkbinhmmlist mkbinhmmlist.exe

distclean:
	$(RM) mkbinhmm.o mkbinhmmlist.o
	$(RM) *~ core
	$(RM) mkbinhmm mkbinhmm.exe
	$(RM) mkbinhmmlist mkbinhmmlist.exe
	$(RM) Makefile
