# Generated automatically from Makefile.in by configure.
# Copyright (c) 1991-2013 Kawahara Lab., Kyoto University
# Copyright (c) 2000-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2013 Julius project team, Nagoya Institute of Technology
# All rights reserved
#
# Makefile.in --- Makefile Template for configure
#
# $Id: Makefile.in,v 1.6 2013/06/20 17:14:16 sumomo Exp $
#
SHELL=/bin/sh

SUBDIRS=libsent libjulius julius mkbingram mkbinhmm adinrec adintool mkgshmm mkss jcontrol gramtools generate-ngram jclient-perl man

CONFIG_SUBDIRS=mkgshmm gramtools jcontrol julius libjulius libsent

RM=/bin/rm -f
prefix=/home/tadan/julius
exec_prefix=${prefix}
INSTALL=/bin/install -c

all:
	for d in $(SUBDIRS); do \
	  (cd $$d; $(MAKE)); \
	done

install:
	for d in $(SUBDIRS); do \
	  (cd $$d; $(MAKE) install); \
	done

install.bin:
	for d in $(SUBDIRS); do \
	  (cd $$d; $(MAKE) install.bin); \
	done

install.man:
	(cd man; $(MAKE) install)

clean:
	for d in $(SUBDIRS); do \
	  (cd $$d; $(MAKE) clean); \
	done
	$(RM) config.log config.cache

distclean:
	for d in $(SUBDIRS); do \
	  if test -f $$d/Makefile; then \
	   (cd $$d; $(MAKE) distclean); \
	  fi; \
	done
	$(RM) config.log config.cache
	$(RM) config.status
	$(RM) Makefile

configure:
	for d in $(CONFIG_SUBDIRS); do \
	  (cd $$d; autoconf); \
	done

doxygen:
	mkdir -p doxygen/en
	mkdir -p doxygen/ja
	cat support/doxygen.conf.en libjulius/doxygen.conf.ver > doxygen/doxygen.conf.en
	cat support/doxygen.conf.ja libjulius/doxygen.conf.ver > doxygen/doxygen.conf.ja
	doxygen doxygen/doxygen.conf.en > doxygen/doxygen-en.log 2> doxygen/doxygen-en-err.log
	doxygen doxygen/doxygen.conf.ja > doxygen/doxygen-ja.log 2> doxygen/doxygen-ja-err.log
