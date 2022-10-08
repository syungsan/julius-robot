#!/usr/bin/perl
# Copyright (c) 1991-2013 Kawahara Lab., Kyoto University
# Copyright (c) 2000-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2013 Julius project team, Nagoya Institute of Technology
#
# mkdfa/mkdfa.pl.  Generated from mkdfa.pl.in by configure. 
#

## setup
$tmpdir = ".";

# mkfa executable location
($thisdir) = ($0 =~ /(.*(\/|\\))[^\/\\]*$/o);
$mkfabin = "${thisdir}mkfa";

# dfa_minimize executable location
$minimizebin = "${thisdir}dfa_minimize";

#############################################################

if ($#ARGV < 0 || $ARGV[0] eq "-h") {
    usage();
}

$make_dict = 1;
$make_term = 1;

$CRLF = 0;

$gramprefix = "";
foreach $arg (@ARGV) {
    if ($arg eq "-t") {
	$make_term = 1;
    } elsif ($arg eq "-n") {
	$make_dict = 0;
    } else {
	$gramprefix = $arg;
    }
}
if ($gramprefix eq "") {
    usage();
}
$gramfile = "$ARGV[$#ARGV].grammar";
$vocafile = "$ARGV[$#ARGV].voca";
$dfafile  = "$ARGV[$#ARGV].dfa";
$fdfafile  = "$ARGV[$#ARGV].dfa.forward";
$dictfile = "$ARGV[$#ARGV].dict";
$termfile = "$ARGV[$#ARGV].term";
$tmpprefix = "$tmpdir/tmp$$";
$tmpgramfile = "${tmpprefix}.grammar";
$tmpvocafile = "${tmpprefix}.voca";
$rgramfile = "${tmpprefix}-rev.grammar";
$tmpheadfile = "${tmpprefix}.h";

# check if input file exists

if (! -f $gramfile) {
    die "cannot open \"$gramfile\"";
}
if (! -f $vocafile) {
    die "cannot open \"$vocafile\"";
}

# sanitize grammar file
open(GRAM,"< $gramfile") || die "cannot open \"$gramfile\"";
open(SGRAM,"> $tmpgramfile") || die "cannot open \"$tmpgramfile\"";
while (<GRAM>) {
    chomp;
    $CRLF = 1 if /\r$/;
    s/\r+$//g;
    s/#.*//g;
    if (/^[ \t]*$/) {
	print SGRAM "\n";
	next;
    }
    print SGRAM "$_\n";
}
close(SGRAM);
close(GRAM);

# generate reverse grammar file
open(GRAM,"< $gramfile") || die "cannot open \"$gramfile\"";
open(RGRAM,"> $rgramfile") || die "cannot open \"$rgramfile\"";
$n = 0;
while (<GRAM>) {
    chomp;
    $CRLF = 1 if /\r$/;
    s/\r+$//g;
    s/#.*//g;
    if (/^[ \t]*$/) {
	print RGRAM "\n";
	next;
    }
    ($left, $right) = split(/\:/);
    if ($CRLF == 1) {
	print RGRAM $left, ': ', join(' ', reverse(split(/ /,$right))), "\r\n";
    } else {
	print RGRAM $left, ': ', join(' ', reverse(split(/ /,$right))), "\n";
    }
    $n ++;
}
close(GRAM);
close(RGRAM);
print "$gramfile has $n rules\n";

# make temporary voca for mkfa (include only category info)
if (! -r $vocafile) {
	die "cannot open voca file $vocafile";
}
open(VOCA,"$vocafile") || die "cannot open vocabulary file";
open(TMPVOCA,"> $tmpvocafile") || die "cannot open temporary file $tmpvocafile";
if ($make_term == 1) {
    open(GTERM, "> $termfile");
}
$n1 = 0;
$n2 = 0;
$termid = 0;
while (<VOCA>) {
    chomp;
    $CRLF = 1 if /\r$/;
    s/\r+$//g;
    s/#.*//g;
    if (/^[ \t]*$/) {
	print TMPVOCA "\n";
	next;
    }
    if (/^%[ \t]*([A-Za-z0-9_]*)/) {
	if ($CRLF == 1) {
	    printf(TMPVOCA "\#%s\r\n", $1);
	} else {
	    printf(TMPVOCA "\#%s\n", $1);
	}
	if ($make_term == 1) {
	    if ($CRLF == 1) {
		printf(GTERM "%d\t%s\r\n",$termid, $1);
	    } else {
		printf(GTERM "%d\t%s\n",$termid, $1);
	    }
	    $termid++;
	}
	$n1++;
    } else {
	$n2++;
    }
}
close(VOCA);
close(TMPVOCA);
if ($make_term == 1) {
    close(GTERM);
}
print "$vocafile    has $n1 categories and $n2 words\n";
print "---\n";

# call mkfa and make .dfa
sub mkfa {
    my ($gram, $voca, $dfa, $h) = @_;
    my $status;
    my $command;
    if ($tmpprefix =~ /cygdrive/) {
	$command = "$mkfabin -e1 -fg `cygpath -w $gram` -fv `cygpath -w $voca` -fo `cygpath -w $dfa` -fh `cygpath -w $h`";
    } else {
	$command = "$mkfabin -e1 -fg $gram -fv $voca -fo $dfa -fh $h";
    }
    print "executing [$command]\n";
    $status = system("$command");
    if ($status != 0) {
	print STDERR "\n";
	print STDERR "**** Error occured in mkfa ***\n";
	print STDERR "*  Temporary files are left in $tmpdir for your debugging. You can delete them manually:\n";
	print STDERR "*            grammar = $tmpgramfile\n";
	print STDERR "*    reverse grammar = $rgramfile\n";
	print STDERR "*     vocab category = $tmpvocafile\n";
	print STDERR "*         header log = $tmpheadfile\n";
	print STDERR "\n";
    }
    return $status;
}

if (! -x $minimizebin) {
    # no minimization
    print "Warning: dfa_minimize not found in the same place as mkdfa.pl\n";
    print "Warning: no minimization performed\n";
    if (&mkfa($rgramfile, $tmpvocafile, $dfafile, $tmpheadfile) != 0) {
	die "stopped";
    }
    if (&mkfa($tmpgramfile, $tmpvocafile, $fdfafile, $tmpheadfile) != 0) {
	die "stopped";
    }
} else {
    # minimize DFA after generation
    if (&mkfa($rgramfile, $tmpvocafile, ${dfafile}.tmp, $tmpheadfile) != 0) {
	die "stopped";
    }
    system("$minimizebin `cygpath -w ${dfafile}.tmp` -o `cygpath -w $dfafile`");
    if (&mkfa($tmpgramfile, $tmpvocafile, ${dfafile}.tmp, $tmpheadfile) != 0) {
	die "stopped";
    }
    system("$minimizebin `cygpath -w ${dfafile}.tmp` -o `cygpath -w $fdfafile`");
    unlink("${dfafile}.tmp");
}

unlink("$tmpgramfile");
unlink("$rgramfile");
unlink("$tmpvocafile");
unlink("$tmpheadfile");
print "---\n";
if ($status != 0) {
    # error
    print "no .dfa or .dict file generated\n";
    exit;
}

# convert .voca -> .dict
# terminal number should be ordered by voca at mkfa output
if ($make_dict == 1) {
    $nowid = -1;
    open(VOCA, "$vocafile")  || die "No vocafile \"$vocafile\" found.\n";
    open(DICT, "> $dictfile") || die "cannot open $dictfile for writing.\n";
    while (<VOCA>) {
	chomp;
	s/\r//g;
	s/#.*//g;
	if (/^[ \t]*$/) {next;}
	if (/^%/) {
	    $nowid++;
	    next;
	} else {
	    @a = split;
	    $name = shift(@a);
	    if ($CRLF == 1) {
		printf(DICT "%d\t[%s]\t%s\r\n", $nowid, $name, join(' ', @a));
	    } else {
		printf(DICT "%d\t[%s]\t%s\n", $nowid, $name, join(' ', @a));
	    }
	}
    }
    close(VOCA);
    close(DICT);
}

$gene = "$dfafile";
if ($make_term == 1) {
    $gene .= " $termfile";
}
if ($make_dict == 1) {
    $gene .= " $dictfile";
}
$gene .= " $fdfafile";

print "generated: $gene\n";

sub usage {
    print "mkdfa.pl --- DFA compiler\n";
    print "usage: $0 [-n] prefix\n";
    print "\t-n ... keep current dict, not generate\n";
    exit;
}
