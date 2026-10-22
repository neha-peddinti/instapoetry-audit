"""
Seed words for the two seed-based style dimensions (Brooke & Hirst 2013).

The original seed lists were not released. These lists follow the paper's
selection criteria and examples, and were drafted for this project.
REVIEW THEM before use: Brooke & Hirst had a second author check every list,
and the induced lexicon is only as good as its seeds. Remove anything that
does not fit the criterion; add words until each list is close to 150.

Criteria (Brooke & Hirst 2013, Sec. 3):
  literary    difficult or archaic words of the kind explained on websites for
              the Bible or Lord of the Rings (e.g. behold, resplendent, amiss, thine)
  colloquial  English slang terms and acronyms (e.g. cuz, gig, asshole, lol)
  subjective  strongly positive and negative terms from a manually built
              sentiment lexicon (Taboada et al. 2011, SO-CAL); loaded from
              data/lexicons/socal in build_style_lexicon.py, not listed here
  objective   emotionally distant alternatives from near-synonym sets
              (e.g. residence for home)

Words that appear in more than one list are removed automatically, as in the paper.
"""

LITERARY = """
behold resplendent amiss thine thee thou thy hath doth ere whence wherefore
hither thither yonder oft nigh alas lo verily perchance betwixt wherein
whereupon henceforth forsooth anon morrow naught aught bequeath beseech
besought smite smote slain wrought forsake forsaken beguile betrothed damsel
maiden steed hearth raiment visage countenance tresses bosom wane sojourn
abode lament woe wretched wroth ire valor fealty liege kinsman thence whither
hitherto unto nay yea wilt shalt dost mayhap belike sooth swain firmament
ethereal effulgent gossamer sylvan verdant vernal crepuscular lambent
ineffable languor reverie quietude requiem elegy dirge threnody lyre zephyr
gloaming eventide nectar ambrosia mirth revelry beholden bestow betoken
enthrall ensnare entreat espy fain forlorn haply lissome palfrey quaff
rapture repast scion sepulchre tempest thrall travail unbidden vale welkin
wend whilst erstwhile bedecked
""".split()

COLLOQUIAL = """
lol lmao lmfao omg btw idk tbh smh wtf brb ikr gonna wanna gotta kinda sorta
dunno gimme lemme ya yeah yep nope nah yup dude bro bruh buddy pal chill dope
legit stuff gig cuz coz lotta bucks cops mom dad okay ok hey wow yay ugh meh
whoa oops haha hehe fam vibe vibes sus lowkey highkey salty slay bae bestie
hangout freaking freakin damn crap sucks sucky dumb weirdo nerd goofy cheesy
cringe hype pumped bummed busted screwed wasted hammered booze snack nap
selfie insta tweet texting junk yummy comfy gross icky kiddo folks gal guy
shit fuck asshole ass bitch dumbass jerk wimp loser cutie hottie sweetie
babe honey awesome cool
""".split()

OBJECTIVE = """
residence individual consume purchase obtain commence terminate assist require
indicate utilize sufficient approximately numerous facility vehicle beverage
garment occupant deceased physician canine feline juvenile adolescent spouse
offspring sibling location structure component procedure process function
operate transport acquire request inquire inform notify depart reside
relocate observe examine evaluate assess quantity frequency duration interval
temperature increase decrease modify adjust maintain retain remove contain
consist comprise establish implement document data information report
statement item material substance compound region sector period method
standard typical moderate relevant adequate additional initial subsequent
prior previous current respectively regarding concerning consequently
accordingly equipment apparatus device mechanism circumstance situation
condition incident occurrence encounter interaction communication
correspondence employment occupation compensation expenditure revenue
""".split()
