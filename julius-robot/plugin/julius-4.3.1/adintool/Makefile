# Generated automatically from Makefile.in by configure.
# Copyright (c) 1991-2013 Kawahara Lab., Kyoto University
# Copyright (c) 2001-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2013 Julius project team, Nagoya Institute of Technology
# All rights reserved 
#
# $Id: Makefile.in,v 1.8 2013/06/20 17:14:16 sumomo Exp $
#
SHELL=/bin/sh
.SUFFIXES:
.SUFFIXES: .c .o
.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ -c $<

LIBSENT=../libsent
LIBJULIUS=../libjulius
CC=gcc
CFLAGS=-O6 -fomit-frame-pointer
CPPFLAGS=-I. -I$(LIBJULIUS)/include -I$(LIBSENT)/include  `$(LIBSENT)/libsent-config --cflags` `$(LIBJULIUS)/libjulius-config --cflags`
LDFLAGS= -L$(LIBJULIUS) `$(LIBJULIUS)/libjulius-config --libs` -L$(LIBSENT) `$(LIBSENT)/libsent-config --libs`
RM=/bin/rm -f
prefix=/home/tadan/julius
exec_prefix=${prefix}
INSTALL=/bin/install -c

############################################################

TARGET=adintool.exe

all: $(TARGET)

$(TARGET): adintool.c $(LIBSENT)/libsent.a $(LIBJULIUS)/libjulius.a
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ adintool.c $(LDFLAGS)

############################################################

install: install.bin

install.bin: $(TARGET)
	${INSTALL} -d ${exec_prefix}/bin
	${INSTALL} $(TARGET) ${exec_prefix}/bin

############################################################

clean:
	$(RM) *.o *~ core
	$(RM) $(TARGET) $(TARGET).exe

distclean:
	$(RM) *.o *~ core
	$(RM) $(TARGET) $(TARGET).exe
	$(RM) Makefile
