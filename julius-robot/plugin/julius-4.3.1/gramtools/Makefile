# Generated automatically from Makefile.in by configure.
# Copyright (c) 1991-2013 Kawahara Lab., Kyoto University
# Copyright (c) 2000-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2013 Julius project team, Nagoya Institute of Technology

# All rights reserved
#
# $Id: Makefile.in,v 1.6 2013/06/20 17:14:17 sumomo Exp $
#
SHELL=/bin/sh
RM=/bin/rm -f
prefix=/home/tadan/julius
exec_prefix=${prefix}
INSTALL=/bin/install -c

SUBDIRS=mkdfa dfa_minimize generate accept_check nextword yomi2voca gram2sapixml dfa_determinize 

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
