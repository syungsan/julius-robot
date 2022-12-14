#!/bin/perl
## Copyright (c) 2002  Takashi Sumiyoshi

# ------------------------------------------------------------
# Julian 形式の文法 (.grammar, .voca) を SAPI XML 文法に変換します。
# 引数なし起動で使い方が表示されます。
# 実行には Jcode モジュールが必要です。
# 出力を UTF-8 形式に変換するのに外部コマンドとして iconv を使用しています。
# ------------------------------------------------------------
# 注意: Julian 形式の文法では、右再帰が使えません。逆に SAPI XML 形式で
# は左再帰が使えません。このツールはその変換まではしないので、左再帰を
# 含む文法は、変換後に手作業で修正する必要があります。
# ------------------------------------------------------------
# 出力される SAPI XML 文法ファイルは、元ファイルの文法の非終端記号、終端記号
# をルールに変換するという単純な変換であるため、よりエレガントにするには
# 手作業で修正する必要があります。
# ------------------------------------------------------------

use strict;
use Jcode;
my $iconv = "iconv -f eucJP -t UTF-8"; # iconv command line

############################################################
# convertphone で使用する
############################################################
sub vowel
{
    if ($_[0] eq "a") { return $_[1];}
    if ($_[0] eq "i") { return $_[2];}
    if ($_[0] eq "u") { return $_[3];}
    if ($_[0] eq "e") { return $_[4];}
    if ($_[0] eq "o") { return $_[5];}
    if ($_[0] eq "a:") { return $_[1]."ー";}
    if ($_[0] eq "i:") { return $_[2]."ー";}
    if ($_[0] eq "u:") { return $_[3]."ー";}
    if ($_[0] eq "e:") { return $_[4]."ー";}
    if ($_[0] eq "o:") { return $_[5]."ー";}
    return 0;
}

############################################################
# サブルーティン: 入力音素配列からカナ文字列を生成。入力は正しいと仮定
############################################################
sub convertphone
{
    my $rval = "";
    my $c;
    my $d;
    my $r;
    while($c = shift @_)
    {
	if ($c eq "k") { $d = shift @_;
	    if ($r = vowel($d,"か","き","く","け","こ")) { $rval .= $r; }
	}
	if ($c eq "ky") { $d = shift @_;
            if ($r = vowel($d,"きゃ","kyi?","きゅ","kye?","きょ")) { $rval .= $r; }
	}
	if ($c eq "s") { $d = shift @_;
	    if ($r = vowel($d,"さ","し","す","せ","そ")) { $rval .= $r; }
	}
	if ($c eq "sy") { $d = shift @_;
            if ($r = vowel($d,"しゃ","syi?","しゅ","しぇ","しょ")) { $rval .= $r; }
	}
	if ($c eq "sh") { $d = shift @_;
	    if ($r = vowel($d,"しゃ","し","しゅ","しぇ","しょ")) { $rval .= $r; }
	}
	if ($c eq "t") { $d = shift @_;
	    if ($r = vowel($d,"た","ち","つ","て","と")) { $rval .= $r; }
	}
	if ($c eq "ts") { $d = shift @_;
	    if ($r = vowel($d,"た","ち","つ","て","と")) { $rval .= $r; }
	}
	if ($c eq "ty") { $d = shift @_;
	    if ($r = vowel($d,"ちゃ","tyi?","ちゅ","ちぇ","ちょ")) { $rval .= $r; }
	}
	if ($c eq "ch") { $d = shift @_;
	    if ($r = vowel($d,"ちゃ","ち","ちゅ","ちぇ","ちょ")) { $rval .= $r; }
	}
	if ($c eq "n") { $d = shift @_;
	    if ($r = vowel($d,"な","に","ぬ","ね","の")) { $rval .= $r; }
	}
	if ($c eq "ny") { $d = shift @_;
	    if ($r = vowel($d,"にゃ","nyi?","にゅ","にぇ","にょ")) { $rval .= $r; }
	}
	if ($c eq "h") { $d = shift @_;
	    if ($r = vowel($d,"は","ひ","ふ","へ","ほ")) { $rval .= $r; }
	}
	if ($c eq "hy") { $d = shift @_;
	    if ($r = vowel($d,"ひゃ","hyi?","ひゅ","ひぇ","ひょ")) { $rval .= $r; }
	}
	if ($c eq "f") { $d = shift @_;
	    if ($r = vowel($d,"は","ひ","ふ","へ","ほ")) { $rval .= $r; }
	}
	if ($c eq "m") { $d = shift @_;
	    if ($r = vowel($d,"ま","み","む","め","も")) { $rval .= $r; }
	}
	if ($c eq "my") { $d = shift @_;
	    if ($r = vowel($d,"みゃ","myi?","みゅ","みぇ","みょ")) { $rval .= $r; }
	}
	if ($c eq "y") { $d = shift @_;
	    if ($r = vowel($d,"や","い","ゆ","え","よ")) { $rval .= $r; }
	}
	if ($c eq "r") { $d = shift @_;
	    if ($r = vowel($d,"ら","り","る","れ","ろ")) { $rval .= $r; }
	}
	if ($c eq "ry") { $d = shift @_;
	    if ($r = vowel($d,"りゃ","ryi?","りゅ","りぇ","りょ")) { $rval .= $r; }
	}
	if ($c eq "w") { $d = shift @_;
	    if ($r = vowel($d,"わ","うぃ","wu?","うぇ","を")) { $rval .= $r; }
	}
	if ($c eq "g") { $d = shift @_;
	    if ($r = vowel($d,"が","ぎ","ぐ","げ","ご")) { $rval .= $r; }
	}
	if ($c eq "gy") { $d = shift @_;
	    if ($r = vowel($d,"ぎゃ","gyi?","ぎゅ","ぎぇ","ぎょ")) { $rval .= $r; }
	}
	if ($c eq "z") { $d = shift @_;
	    if ($r = vowel($d,"ざ","じ","ず","ぜ","ぞ")) { $rval .= $r; }
	}
	if ($c eq "zy") { $d = shift @_;
	    if ($r = vowel($d,"じゃ","zyi?","じゅ","じぇ","じょ")) { $rval .= $r; }
	}
	if ($c eq "j") { $d = shift @_;
	    if ($r = vowel($d,"じゃ","じ","じゅ","じぇ","じょ")) { $rval .= $r; }
	}
	if ($c eq "d") { $d = shift @_;
	    if ($r = vowel($d,"だ","ぢ","づ","で","ど")) { $rval .= $r; }
	}
	if ($c eq "dy") { $d = shift @_;
	    if ($r = vowel($d,"ぢゃ","dyi?","ぢゅ","ぢぇ","ぢょ")) { $rval .= $r; }
	}
	if ($c eq "b") { $d = shift @_;
	    if ($r = vowel($d,"ば","び","ぶ","べ","ぼ")) { $rval .= $r; }
	}
	if ($c eq "by") { $d = shift @_;
	    if ($r = vowel($d,"びゃ","byi?","びゅ","びぇ","びょ")) { $rval .= $r; }
	}
	if ($c eq "p") { $d = shift @_;
	    if ($r = vowel($d,"ぱ","ぴ","ぷ","ぺ","ぽ")) { $rval .= $r; }
	}
	if ($c eq "py") { $d = shift @_;
	    if ($r = vowel($d,"ぴゃ","pyi?","ぴゅ","ぴぇ","ぴょ")) { $rval .= $r; }
	}
	if ($c eq "N") { $rval .= "ん" }
	if ($c eq "q") { $rval .= "っ" }
	if ($c eq "sp") { $rval .= '@sp' }
	if ($c eq "silB") { $rval .= '@silB' }
	if ($c eq "silE") { $rval .= '@silE' }

	if ($r = vowel($c,"あ","い","う","え","お")) { $rval .= $r; }
    }
    return $rval;
}

############################################################
# メイン関数
############################################################
if (@ARGV == 0)
{
    print STDERR << "EOF";
gram2sapixml.pl   by Takashi Sumiyoshi 2002
usage: gram2sapixml.pl [basename] ...

   input files: <basename>.grammar (Julian grammar file)
                <basename>.voca    (Julian voca file)
   output file: <basename>.xml     (SAPI Grammar XML file in UTF-8 Format)

This script uses the iconv command to convert the encoding.
EOF
    exit;
}

my $removesps = 1;   # sp, silB, silE を除く

while(@ARGV)
{
    my $filebase = shift @ARGV;
    my $grammarfile = $filebase . ".grammar";
    my $vocafile = $filebase . ".voca";
    my $sapixmlfile = $filebase . ".xml";

    print STDERR "Processing $vocafile, $grammarfile...\n";

    my $vocaword = "";
    my %lexicon_disp;
    my %lexicon_yomi;
    my %grammar_left;
    my @input;
    my $disp;
    my $yomi;
    my $hiragana;

    ###
    ### load voca file
    ###
    open (VOCA, $vocafile) or die "Cannot open $vocafile";

    while(<VOCA>) {
	chomp;
	next if /^#/;
	@input = split (/[ \t]+/, $_);
	if (/^\%/) {
	    s/#.*$//;
	    $vocaword = substr($_, 1); # 先頭の % を抜く
	    $vocaword =~ s/^[ \t]+//g;
	    $vocaword =~ s/[ \t]+$//g;
	} else {
	    $disp = shift @input;
	    $disp = Jcode->new($disp)->euc;
	    if ($disp ne "")
	    {
		if ($removesps == 1 && ($disp eq "sp" || $disp eq "silB" || $disp eq "silE")) {
		} else {
		    # 音素表記をかな文字列に変換
		    $hiragana = convertphone(@input);
#		print "voca [$vocaword] in $disp,$hiragana\n";
		    
		    # lexicon_disp, lexicon_yomi に格納
		    push @{$lexicon_disp{$vocaword}}, $disp;
		    push @{$lexicon_yomi{$vocaword}}, $hiragana;
		}
	    }
	}
    }
    close (VOCA);

    ###
    ### load grammar file
    ###
    open (GRAMMAR, $grammarfile) or die "Cannot open $grammarfile";

    my $left;
    while(<GRAMMAR>)
    {
	chomp;
	next if /^#/;
	s/#.*$//;
	next if $_ eq "";
	@input = split (/[ \t:]+/, $_);
	$left = shift @input;

	# grammar_left は配列へのリファレンスの配列を要素にもつ連想配列
	# MEMO: [@input] を \@input とかすると実体がすべて同じになりまずい
	push @{$grammar_left{$left}}, [@input];
    }

    close (GRAMMAR);

    ###
    ### save sapixml file
    ###

    ###
    ### convert by iconv
    ###
    open (SAPIXML, "| $iconv > $sapixmlfile") or die "Cannot open $sapixmlfile or cannot exec iconv";

    print SAPIXML "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    print SAPIXML "<GRAMMAR>\n";

    # まずは grammar ファイルのルール
    # RULEREF を並べる
    my $i;
    my $n;
    my $a;
    my @b;
    foreach $i (keys %grammar_left)
    {
	if ($i eq "S")
	{
	    print SAPIXML "<RULE name=\"$i\" toplevel=\"ACTIVE\">\n";
	} else {
	    print SAPIXML "<RULE name=\"$i\" toplevel=\"INACTIVE\">\n";
	}
	print SAPIXML "  <L>\n";
	while ($a = shift @{$grammar_left{$i}})
	{
	    print SAPIXML "    <P>\n";
	    @b = @{$a};
	    while ($n = shift @b)
	    {
		if ($removesps == 1 && ! exists $lexicon_disp{$n} && ! exists $grammar_left{$n})
		{
#		    print SAPIXML "#     <RULEREF name=\"$n\"/>\n";
		} else {
		    print SAPIXML "      <RULEREF name=\"$n\"/>\n";
		}
	    }
	    print SAPIXML "    </P>\n";
	}
	
	print SAPIXML "  </L>\n";
	print SAPIXML "</RULE>\n";
    }

    # そして voca ファイルのカテゴリ名→単語
    foreach $i (keys %lexicon_disp)
    {
	print SAPIXML "<RULE name=\"$i\" toplevel=\"INACTIVE\">\n";
	print SAPIXML "  <L>\n";
	while ($disp = shift @{$lexicon_disp{$i}})
	{
	    $yomi = shift @{$lexicon_yomi{$i}};
	    if ($disp eq $yomi)
	    {
		print SAPIXML "    <P>$yomi</P>\n";
	    } else {
		print SAPIXML "    <P>/$disp/$yomi;</P>\n";
	    }
	}
	print SAPIXML "  </L>\n";
	print SAPIXML "</RULE>\n";
    }
    print SAPIXML "</GRAMMAR>\n";
}
