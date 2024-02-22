import re
from nltk import sent_tokenize

ALL = re.compile('catalyst|sensitizer|A:D|D:A|donor|acceptor', flags=re.I)
ALL_ = re.compile('catalyst|A:D|D:A|donor|acceptor', flags=re.I)
filter_mat = re.compile('electrostatic interactions|photosensitisation|BHJ-SCs|BHJs|light absorber|NP-OPV|organic solar|n-type'
                        '|different sensitizers|adsorbed molecules|blend films|bilayer|π‐conjugated polymers|π-conjugated donor-acceptor|'
                        'p–i–n-type|fullerene-free|polymeric|ηIQE|SM-donors|donor-bridge-acceptor|derivatives|composites'
                        'photoelectron-transfer|electron transfer|block copolymer|copolymer|D-π-A|A-DA|A–D–A|\(D–A\)|D-A|D:A'
                        '|counterion present|Grignard reagents|acceptor molecules|electron-deficient|organic photovoltaic'
                        '|oxidized catalyst|heterogeneous photocatalysts|quantum dots|catalyst modules|OSC-based'
                        '|transition-metal complexes|oxidized Photosensitizer|metal-based|all‐PSCs|π-bridge'
                        '|dyes|nonsacrificial electron donors|binuclear unit with|sacrificial electron donors'
                        '|oxidized sensitizer|sacrificial electron donor|oxidized donor|p-type|brominated|A‐D‐A'
                        '|water oxidation catalyst|electroactive units|photocatalytic system|A-A-D structured'
                        '|triplet photosensitizers|various materials|homogeneous photochemical CO2 reduction catalysts|'
                        'chromophore-catalyst assemblies|different triplet acceptors|different triplet acceptor|different triplet accept'
                        '|proposed triplet acceptor|lower energy|proposed triplet accept|water oxidation catalysis|'
                        'triplet energy acceptors|triplet energy acceptor|triplet energy accept|Triplet acceptors|Triplet acceptor'
                        '|Triplet accept|triplet photosensitizer|triplet energy|A-D-A-structured|boron-containing'
                        '|positive charge accumulator|its variant|photocatalytic reac|spectra of the acceptor|requisite acceptor'
                        '|requisite accept|metal complex photocatalyst|components|metal complexes|and/or its derivatives'
                        '|highly electron-rich donor|multichoromophoric design principle|electron-donor|electron-acceptors'
                        '|electron-donor units|n-electron|electron-acceptor|electron-donor unit|internal acceptor|photosensitizers, dyads'
                        '|multicomponent system|related bidentate lig|its analogues|method of preparation|semiconductor composites'
                        '|Photocatalytic|bis\(donor\)-acceptor|donor-bis\(acceptor\)|pi-conjugated|non-fullerene-acceptor-based'
                        '|dye molecules|metal phosphines|first-row transition metal catalysts|pyridine and amine derivatives'
                        '|second- and third-row transition metal polypyridines|molecular catalyst|metal clusters|Visible-Light Phot'
                        '|phthalocyanines|electron acceptors|electron acceptor|electron accept|moieties|photocatalysts|donor−acceptor'
                        '|chromophore−catalyst assemblies|nonsacrificial donor|different sacrificial donor|reduction catalysts'
                        '|new|semiconductor photocatalysts|gap semiconductor|gap semiconduct|conduction band region|semiconductors'
                        '|semiconductor|semiconduct|noble metal|transition-metal oxide|oxidized sensitizers unit|oxidized sensitizer unit'
                        '|precious-metal free|singlet energy accepter|our catalyst family|'
                        'hybrid photocatalytic system|DCSQ|quinoxaline-based|BHJ-based|derivative'
                        '|hybrid particles|materials|material|methylated|fluorinated|nonfullerene|A–D'
                        'visible-light-harvesting antenna|surface defects|triplet sensitizers|SM‐donors'
                        '|photoactivated don|hybrid systems|ground state acceptor|ground state acceptors|light-absorbing'
                        '|bing photosensitizer|Visible-light Phot|dual phot|photocatalyst|photoactive|appropriate photosensitizer'
                        '|organic photosensitizer|ganic photosensitizer|visible-light|two electron accept|two electron acceptor'
                        '|two cyanoacrylic acids|electron spacers|four new dyes|carbocation intermediate|dye-adsorbed|dye-sensitized'
                        '|photolatent-base|visible-light sensitizer|supramolecular photocatalyst|transition metal complexes|metal complex'
                        '|panchromatic photosensitizer|metal-complex|mate buffer|polymeric photosensitizer'
                        '|fluorophore-derived|terpolymer|semiconductor-based'
                        '|conjugated polymer|singlet sensitizers|\(NFA\)-based'
                        '|molecular|Electrochemical|transition-metal catalysts|secondary photosensitizer|adsorption-photocatalysts|'
                        'earth-abundant|metal oxide|framew|chiral phosph|catalytic system|bing molecule|'
                        'Metal oxides|pulling electrons|various electron donor|visible light'
                        '|cocatalysts|various don|energy acceptor|energy accept|sacrificial donors|sacrificial acceptors'
                        '|sacrificial acceptor|catalyst hybrids|efficient catalysts|triplet exciton|triplet acceptors|triplet acceptor'
                        '|unknown|electron-transfer-catalytic|titled|title|D–π–A|donor–acceptor'
                        'electron-transfer|region|linkers|linker|e-derived|sacrificial|metallic|electrocatalytic|one-|dye molecule'
                        '|oxidized states|excited-state|radiation|emitters|emitter|particle states'
                        '|energy gap|singlet energy|energy|photocatalysis|catalysis|sensitizers|sensitizer'
                        '|single Molecules|accepts|acceptors|acceptor|accept|photosensitizers|photosensitizer|donors|donor'
                        '|electrocatalysts|catalysts|catalyst|unspecified|modular|laccase|solvents|solvent|triplet|catalytic'
                        '|compact|solutions|solution|homogeneous|inorganic|organic|multielectron|electron|coordinated|concentration'
                        '|excited|reduction|various|vesicles|reduced|secondary|nanoparticles|nanoparticle|dianionic|monoanionic dye'
                        '|monoanionic|polymers|polymer|cationic|supramolecular systems|supramolecular|mononuclear|Multinuclear|macrocycle'
                        '|molecules|metal ions|metal ion|bacteria|compounds|python|Chemicals|OPVs|OPV|OSCs|OSC|DSSCs|DSSC'
                        '|PSCs|Eg|NFA|JSC|VOC|PCE|EQEEL|HOMO|LUMO|solar cells|bulk heterojunction|\(BHJ\)|Solar Cell|solubilizing|'
                        'magnetic|cation|substrate|films|D–A co|D–A|D/A|\(D\)|\(A\)|layers|layer|photon-to-charge conversion efficiency|'
                        'Pair State|small molecule|multilayer|hole transport chromophore|room temperature|photovoltaics|groups|group'
                        '|parts|part|halogenation|EHOMO|ELUMO|fullerene-based|BHJ|conjugated|p-i-n|'
                        'A–D–A\)-type|fluorinated non-fullerene|non‐fullerene|non-fullerene|neat|fullerene‐based|fullerene‐derived|fullerene|small'
                        'π-bridge-|-π-|chlorinated|PDOS|small-molecule|triazine-based|small|all|transition|A1-D-A2 type|π−bridge', re.I)



def filter_file(context, threshold=4):
    count = 0
    for sen in sent_tokenize(context):
        if ALL_.findall(sen):
            count += 1
    if count > threshold:
        return True
    else:
        return False


def filter_material(word):
    # 过滤不属于材料的类型
    word = filter_mat.sub('', word)
    return word


def find_parent_child_strings(lst):
    lst = [w.strip(' ') for w in lst]
    lst = sorted(lst, key = lambda x: len(x), reverse = True)
    lst = list(set(lst))
    results = []
    visted = []
    for i in range(len(lst)):
        temp = []
        for j in range(i + 1, len(lst)):
            if is_parent_child(lst[i], lst[j]):
                temp.append(lst[j])
                visted.append(lst[j])
        if lst[i] in visted:
            continue
        else:
            temp.insert(0, lst[i])
            temp = list(set(temp))
            temp = sorted(temp, key=lambda x: len(x), reverse=True)
            results.append(temp)
    return results


def is_parent_child(string1, string2):
    string1 = string1.replace(' ', '')
    string2 = string2.replace(' ', '')
    if string1 in string2 or string2 in string1:
        return True
    else:
        return False


if __name__ == '__main__':
    string = ['PC71BM', 'PTB7', 'DIo', 'PTB7', 'PC71BM']
    print(find_parent_child_strings(string))