"""
Perform Cochran-Mantel-Haenszel chi-squared tests on stratified contingency tables.

Each stratum is a population's contingency table; each population has a case and a control.

Each contingency table is 2x2 - case and control x REF and ALT allele counts.

ALT and REF allele counts are calculated by multiplying the ploidy of the population by ...
... either the ALT freq or (1-ALT_freq), for each of case and control - unless any of ...
... the counts are np.nan, then skip population.

TODO: allow user to select specific populations (whichpops) for get_ploidy()
"""
import os, sys, argparse, shutil, subprocess, pandas as pd, threading, ipyparallel, time
import pickle
from os import path as op


def check_pyversion() -> None:
    """Make sure python is 3.6 <= version < 3.8."""
    pyversion = float(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))
    if not pyversion >= 3.6:
        text = f'''FAIL: You are using python {pyversion}. This pipeline was built with python 3.7.
FAIL: use  3.6 <= python version < 3.8
FAIL: exiting cmh_test.py'''
        print(ColorText(text).fail())
        exit()
    if not pyversion < 3.8:
        print(ColorText("FAIL: python 3.8 has issues with the ipyparallel engine returns.").fail())
        print(ColorText("FAIL: use  3.6 <= python version < 3.8").fail())
        print(ColorText("FAIL: exiting cmh_test.py").fail())
        exit()


def pklload(path:str):
    """Load object from a .pkl file."""
    pkl = pickle.load(open(path, 'rb'))
    return pkl


def get_client(profile='default') -> tuple:
    """Get lview,dview from ipcluster."""
    rc = ipyparallel.Client(profile=profile)
    dview = rc[:]
    lview = rc.load_balanced_view()

    return lview, dview


def attach_data(**kwargs) -> None:
    """Load object to engines."""
    import time

    num_engines = len(kwargs['dview'])
    print(ColorText("\nAdding data to engines ...").bold())
    print(ColorText("\tWARN: Watch available mem in another terminal window: 'watch free -h'").warn())
    print(ColorText("\tWARN: If available mem gets too low, kill engines and restart cmh_test.py with fewer engines: 'ipcluster stop'").warn())
    for key,value in kwargs.items():
        if key != 'dview':
            print(f'\tLoading {key} ({value.__class__.__name__}) to {num_engines} engines')
            kwargs['dview'][key] = value
            time.sleep(1)
    time.sleep(10)

    return None


def watch_async(jobs:list, phase=None) -> None:
    """Wait until jobs are done executing, show progress bar."""
    from tqdm import trange

    print(ColorText(f"\nWatching {len(jobs)} {phase} jobs ...").bold())
    
    job_idx = list(range(len(jobs)))
    for i in trange(len(jobs)):
        count = 0
        while count < (i+1):
            count = len(jobs) - len(job_idx)
            for j in job_idx:
                if jobs[j].ready():
                    count += 1
                    job_idx.remove(j)
    pass


class ColorText():
    """
    Use ANSI escape sequences to print colors +/- bold/underline to bash terminal.
    """
    def __init__(self, text:str):
        self.text = text
        self.ending = '\033[0m'
        self.colors = []

    def __str__(self):
        return self.text

    def bold(self):
        self.text = '\033[1m' + self.text + self.ending
        return self

    def underline(self):
        self.text = '\033[4m' + self.text + self.ending
        return self

    def green(self):
        self.text = '\033[92m' + self.text + self.ending
        self.colors.append('green')
        return self

    def purple(self):
        self.text = '\033[95m' + self.text + self.ending
        self.colors.append('purple')
        return self

    def blue(self):
        self.text = '\033[94m' + self.text + self.ending
        self.colors.append('blue')
        return self

    def warn(self):
        self.text = '\033[93m' + self.text + self.ending
        self.colors.append('yellow')
        return self

    def fail(self):
        self.text = '\033[91m' + self.text + self.ending
        self.colors.append('red')
        return self
    pass


def askforinput(msg='Do you want to proceed?', tab='', newline='\n'):
    """Ask for input; if msg is default and input is no, exit."""
    while True:
        inp = input(ColorText(f"{newline}{tab}INPUT NEEDED: {msg} \n{tab}(yes | no): ").warn().__str__()).lower()
        if inp in ['yes', 'no']:
            if inp == 'no' and msg=='Do you want to proceed?':
                print(ColorText('exiting %s' % sys.argv[0]).fail())
                exit()
            break
        else:
            print(ColorText("Please respond with 'yes' or 'no'").fail())
    return inp


def wait_for_engines(engines:int, profile:str):
    """Reload engines until number matches input engines arg."""
    lview = []
    dview = []
    count = 1
    while any([len(lview) != engines, len(dview) != engines]):
        if count % 30 == 0:
            # if waiting too long..
            # TODO: if found engines = 0, no reason to ask, if they continue it will fail
            print('count = ', count)
            print(ColorText("\tFAIL: Waited too long for engines.").fail())
            print(ColorText("\tFAIL: Make sure that if any cluster is running, the -e arg matches the number of engines.").fail())
            print(ColorText("\tFAIL: In some cases, not all expected engines can start on a busy server.").fail())
            print(ColorText("\tFAIL: Therefore, it may be the case that available engines will be less than requested.").fail())
            print(ColorText("\tFAIL: cmh_test.py found %s engines, with -e set to %s" % (len(lview), engines)).fail())
            answer = askforinput(msg='Would you like to continue with %s engines? (choosing no will wait another 60 seconds)' % len(lview), tab='\t', newline='')
            if answer == 'yes':
                break
        try:
            lview,dview = get_client(profile=profile)
        except (OSError, ipyparallel.error.NoEnginesRegistered, ipyparallel.error.TimeoutError):
            lview = []
            dview = []
        time.sleep(2)
        count += 1

    print('\tReturning lview,dview (%s engines) ...' % len(lview))

    return lview,dview


def launch_engines(engines:int, profile:str):
    """Launch ipcluster with engines under profile."""
    print(ColorText(f"\nLaunching ipcluster with {engines} engines...").bold())

    def _launch(engines, profile):
        subprocess.call([shutil.which('ipcluster'), 'start', '-n', str(engines), '--daemonize'])

    # first see if a cluster has already been started
    started = False
    try:
        print("\tLooking for existing engines ...")
        lview,dview = get_client(profile=profile)
        if len(lview) != engines:
            lview,dview = wait_for_engines(engines, profile)
        started = True
    except (OSError, ipyparallel.error.NoEnginesRegistered, ipyparallel.error.TimeoutError):
        print("\tNo engines found ...")

    # if not, launch 'em
    if started is False:
        print("\tLaunching engines ...")
        # pid = subprocess.Popen([shutil.which('ipcluster'), 'start', '-n', str(engines)]).pid
        x = threading.Thread(target=_launch, args=(engines,profile,), daemon=True)
        x.daemon=True
        x.start()
        lview,dview = wait_for_engines(engines, profile)

    return lview,dview


def get_freq(string:str) -> float:
    """Convert VarScan FREQ to floating decimal [0,1]."""
    import numpy
    try:
        freq = float(string.replace("%", "")) / 100
    except AttributeError as e:
        # if string is np.nan
        freq = numpy.nan
    return freq


def get_table(casedata, controldata, locus):
    """Create stratified contingency tables (each 2x2) for a given locus.

    Each stratum is a population.
    
    Contingency table has treatment (case or control) as rows, and
        allele (REF or ALT) as columns.

    Example table
    -------------
    # in python
    [1] mat = np.asarray([[0, 6, 0, 5],
                          [3, 3, 0, 6],
                          [6, 0, 2, 4],
                          [5, 1, 6, 0],
                          [2, 0, 5, 0]])
    [2] [np.reshape(x.tolist(), (2, 2)) for x in mat]
    
    [out]
        [array([[0, 6],
                [0, 5]]),
         array([[3, 3],
                [0, 6]]),
         array([[6, 0],
                [2, 4]]),
         array([[5, 1],
                [6, 0]]),
         array([[2, 0],
                [5, 0]])]

    # from R - see https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/mantelhaen.test
    c(0, 0, 6, 5,
      ...)
            Response
    Delay  Cured Died
    None     0    6
    1.5h     0    5
    ...

    """
    import numpy, pandas
    tables = []  # - a list of lists
    for casecol,controlcol in pairs.items():
        # get ploidy of pop
        pop = casecol.split('.FREQ')[0]
        pop_ploidy = ploidy[pop]
        
        # get case-control frequencies of ALT allele
        case_freq = get_freq(casedata.loc[locus, casecol])
        cntrl_freq = get_freq(controldata.loc[locus, controlcol])
        
        # see if either freq is np.nan, if so, skip this pop
        if sum([x!=x for x in [case_freq, cntrl_freq]]) > 0:
            continue
        
        # collate info for locus (create contingency table data)
        t = []
        for freq in [cntrl_freq, case_freq]:
            t.extend([(1-freq)*pop_ploidy,
                      freq*pop_ploidy])
        tables.append(t)
    # return contingency tables (elements of list) for this locus stratified by population (list index)
    return [numpy.reshape(x.tolist(), (2, 2)) for x in numpy.asarray(tables)]


def create_tables(*args):
    """Get stratified contingency tables for all loci in cmh_test.py input file."""
    import pandas
    tables = {}
    for locus in args[0].index:
        tables[locus] = get_table(*args, locus)
    return tables    


def cmh_test(*args):
    """Perform Cochran-Mantel-Haenszel chi-squared test on stratified contingency tables."""
    import pandas
    from statsmodels.stats.contingency_tables import StratifiedTable as cmh
    
    tables = create_tables(*args)
    
    # fill in a dataframe with cmh test results, one locus at a time
    results = pandas.DataFrame(columns=['locus', 'odds_ratio', 'p-value',
                                        'lower_confidence', 'upper_confidence', 'num_pops'])
    for locus,table in tables.items():
        # cmh results for stratified contingency tables (called "table" = an array of tables)
        cmh_res = cmh(table)
        res = cmh_res.test_null_odds(True)  # statistic and p-value
        odds_ratio = cmh_res.oddsratio_pooled  # odds ratio
        conf = cmh_res.oddsratio_pooled_confint()  # lower and upper confidence
        results.loc[len(results.index), :] = (locus, odds_ratio, res.pvalue, *conf, len(table))
    
    return results


def parallelize_cmh(casedata, controldata, lview):
    """Parallelize Cochran-Mantel-Haenszel chi-squared tests by groups of loci."""
    print(ColorText('\nParallelizing CMH calls ...').bold())
    import math, tqdm, pandas
    
    jobsize = math.ceil(len(casedata.index)/len(lview))
    
    # send jobs to engines
    numjobs = (len(casedata.index)/jobsize)+1
    print(ColorText("\nSending %d jobs to engines ..." % numjobs ).bold())
    jobs = []
    loci_to_send = []
    count = 0
    for locus in tqdm.tqdm(casedata.index):
        count += 1
        loci_to_send.append(locus)
        if len(loci_to_send) == jobsize or count == len(casedata.index):
            jobs.append(lview.apply_async(cmh_test, *(casedata.loc[loci_to_send, :],
                                                      controldata.loc[loci_to_send, :])))
#             jobs.append(cmh_test(casedata.loc[loci_to_send, :],
#                                  controldata.loc[loci_to_send, :]))  # for testing
            loci_to_send = []

    # wait until jobs finish
    watch_async(jobs, phase='CMH test')
    
    # gather output, concatenate into one datafram
    print(ColorText('\nGathering parallelized results ...').bold())
    output = pandas.concat([j.r for j in jobs])
#     output = pandas.concat([j for j in jobs])  # for testing
    
    return output


def get_cc_pairs(casecols, controlcols, case, control):
    """For a given population, pair its case column with its control column."""
    badcols = []
#     global pairs  # for debugging
    pairs = {}
    for casecol in casecols:
        controlcol = casecol.replace(case, control)
        if not controlcol in controlcols:
            badcols.append((casecol, controlcol))
            continue
        pairs[casecol] = controlcol
    
    if len(badcols) > 0:
        print(ColorText('FAIL: The following case populations to not have a valid control column in dataframe.').fail())
        for cs,ct in badcols:
            print(ColorText(f'FAIL: no match for {cs} named {ct} in dataframe').fail())
        print(ColorText('FAIL: These case columns have not been paired and will be excluded from analyses.').fail())
        askforinput()
    
    return pairs    


def get_data(df, case, control):
    """Separate input dataframe into case-only and control-only dataframes."""
    # get columns for case and control
    casecols = [col for col in df if case in col and 'FREQ' in col]
    cntrlcols = [col for col in df if control in col and 'FREQ' in col]
    
    # isolate data to separate dfs
    casedata = df[casecols]
    controldata = df[cntrlcols]
    assert casedata.shape == controldata.shape
    
    # pair up case-control pops
    pairs = get_cc_pairs(casecols, cntrlcols, case, control)
    
    return casedata, controldata, pairs

def get_parse():
    """
    Parse input flags.
    # TODO check arg descriptions, and if they're actually used.
    """
    parser = argparse.ArgumentParser(description=print(mytext),
                                     add_help=True,
                                     formatter_class=argparse.RawTextHelpFormatter)
    requiredNAMED = parser.add_argument_group('required arguments')
    
    requiredNAMED.add_argument("-i", "--input",
                               required=True,
                               default=None,
                               dest="input",
                               type=str,
                               help='''/path/to/VariantsToTable_output.txt
It is assumed that there is either a 'locus' or 'unstitched_locus' column.
The 'locus' column elements are the hyphen-separated
CHROM-POS. If the 'unstitched_chrom' column is present, the code will use the
'unstitched_locus' column for SNP names, otherwise 'CHROM' and 'locus'. The
'unstitched_locus' elements are therefore the hyphen-separated 
unstitched_locus-unstitched_pos. FREQ columns from VarScan are also 
assumed.
''')
    requiredNAMED.add_argument("-o","--outdir",
                               required=True,
                               default=None,
                               dest="outdir",
                               type=str,
                               help='''/path/to/cmh_test_output_dir/
File output from cmh_test.py will be saved in the outdir, with the original
name of the input file, but with the suffix "_CMH-test-results.txt"''')
    requiredNAMED.add_argument("--case",
                               required=True,
                               default=None,
                               dest="case",
                               type=str,
                               help='''The string present in every column for pools in "case" treatments.''')
    requiredNAMED.add_argument("--control",
                               required=True,
                               default=None,
                               dest="control",
                               type=str,
                               help='''The string present in every column for pools in "control" treatments.''')    
    requiredNAMED.add_argument("-p","--ploidy",
                               required=True,
                               default=None,
                               dest="ploidyfile",
                               type=str,
                               help='''/path/to/the/ploidy.pkl file output by the VarScan pipeline. This is a python
dictionary with key=pool_name, value=dict with key=pop, value=ploidy. The code
will prompt for pool_name if necessary.''')
    requiredNAMED.add_argument("-e","--engines",
                               required=True,
                               default=None,
                               dest="engines",
                               type=int,
                               help="The number of ipcluster engines that will be launched.")
    parser.add_argument("--ipcluster-profile",
                        required=False,
                        default='default',
                        dest="profile",
                        type=str,
                        help="The ipcluster profile name with which to start engines. Default: 'default'")
    parser.add_argument('--keep-engines',
                        required=False,
                        action='store_true',
                        dest="keep_engines",
                        help='''Boolean: true if used, false otherwise. If you want to keep
the ipcluster engines alive, use this flag. Otherwise engines will be killed automatically.
(default: False)''')

    # check flags
    args = parser.parse_args()
    if not op.exists(args.outdir):
        print(ColorText(f"FAIL: the directory for the output file(s) does not exist.").fail())
        print(ColorText(f"FAIL: please create this directory: %s" % args.outdir).fail())
        print(ColorText("exiting cmh_test.py").fail())
        exit()

    # make sure input and ploidyfile exist
    nopath = []
    for x in [args.input, args.ploidyfile]:  # TODO: check for $HOME or other bash vars in path
        if not op.exists(x):
            nopath.append(x)

    # if input or ploidy file do not exist:
    if len(nopath) > 0:
        print(ColorText("FAIL: The following path(s) do not exist:").fail())
        for f in nopath:
            print(ColorText("\tFAIL: %s" % f).fail())
        print(ColorText('\nexiting cmh_test.py').fail())
        exit()
    
    print('args = ', args)
    return args


def choose_pool(ploidy:dict) -> dict:
    """Choose which the pool to use as a key to the ploidy dict."""
    keys = list(ploidy.keys())
    if len(keys) == 1:
        # return the value of the dict using the only key
        return ploidy[keys[0]]

    print(ColorText('\nPlease choose a pool that contains the population of interest.').bold())
    nums = []
    for i,pool in enumerate(keys):
        print('\t%s %s' % (i, pool))
        nums.append(i)

    while True:
        inp = int(input(ColorText("\tINPUT NEEDED: Choose file by number: ").warn()).lower())
        if inp in nums:
            pool = keys[inp]
            break
        else:
            print(ColorText("\tPlease respond with a number from above.").fail())

    # make sure they've chosen at least one account
    while pool is None:
        print(ColorText("\tFAIL: You need to specify at least one pool. Revisiting options...").fail())
        pool = choose_pool(ploidy, args, keep=None)

    return ploidy[pool]


def get_ploidy(ploidyfile) -> dict:
    """Get the ploidy of the populations of interest, reduce ploidy pkl."""
    print(ColorText('\nLoading ploidy information ...').bold())
    # have user choose key to dict
    return choose_pool(pklload(ploidyfile))


def read_input(inputfile):
    """Read in inputfile, set index to locus names."""
    print(ColorText('\nReading input file ...').bold())
    # read in datatable
    df = pd.read_table(inputfile, sep='\t')
    
    # set df index
    locuscol = 'unstitched_locus' if 'unstitched_locus' in df.columns else 'locus'
    if locuscol not in df:
        print(ColorText('\nFAIL: There must be a column for locus IDs - either "unstitched_locus" or "locus"').fail())
        print(ColorText('FAIL: The column is the hyphen-separated CHROM and POS.').fail())
        print(ColorText('exiting cmh_test.py').fail())
        exit()
    df.index = df[locuscol].tolist()
    
    return df


def main():
    # make sure it's not python3.8
    check_pyversion()

    # parse input arguments
    args = get_parse()
    
    # read in datatable
    df = read_input(args.input)

    # get ploidy for each pool to use to correct read counts for pseudoreplication
#     global ploidy  # for debugging
    ploidy = get_ploidy(args.ploidyfile)
    
    # isolate case/control data
    casedata, controldata, pairs = get_data(df, args.case, args.control)
    
    # get ipcluster engines
    lview,dview = launch_engines(args.engines, args.profile)
    
    # attach data and functions to engines
    attach_data(ploidy=ploidy,
                case=args.case,
                control=args.control,
                pairs=pairs,
                cmh_test=cmh_test,
                get_freq=get_freq,
                get_table=get_table,
                create_tables=create_tables,
                dview=dview)
    
    # run cmh tests in parallel
    output = parallelize_cmh(casedata, controldata, lview)
    
    # write to outfile
    outfile = op.join(args.outdir, op.basename(args.input).split(".")[0] + '_CMH-test-results.txt')
    print(ColorText(f'\nWriting all results to: ').bold().__str__()+ f'{outfile} ...')
    output.to_csv(outfile,
                  sep='\t', index=False)

    # kill ipcluster to avoid mem problems
    if args.keep_engines is False:
        print(ColorText("\nStopping ipcluster ...").bold())
        subprocess.call([shutil.which('ipcluster'), 'stop'])
    
    print(ColorText('\nDONE!!\n').green().bold())
    pass


if __name__ == '__main__':
    mytext = ColorText('''
*****************************************************************************
                                CoAdapTree's
                                
      ______  __      ___  __    __     ________                    _
    |  ____| |   \\  /   | |  |  |  |   |__   __| ____     _____  __| |__
    | |      |    \\/    | |  |__|  |      | |   / __ \\  | ____| |__   __|
    | |      |  |\\  /|  | |   __   |      | |  | /__\\_| |___       | |
    | |____  |  | \\/ |  | |  |  |  |      | |  | \____   ___| |    | |
    |______| |__|    |__| |__|  |__|      |_|   \\____/  |_____|    |_|


                   Cochran-Mantel-Haenszel chi-squared test   
*****************************************************************************''').green().bold().__str__()
    main()


