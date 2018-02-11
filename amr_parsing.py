#!/usr/bin/python
from __future__ import absolute_import
import sys
import codecs
import time
import string
import re
import random
import argparse
import cPickle as pickle
from common.SpanGraph import *
from common.AMRGraph import *
import subprocess
from Aligner import *
from parser import *
from model import Model
from preprocessing import *
import constants
from graphstate import GraphState
from smatch.api import SmatchScorer
reload(sys)
sys.setdefaultencoding('utf-8')

log = sys.stderr
LOGGED = False
experiment_log = sys.stdout


def get_dependency_graph(stp_dep, from_file=False):
    if from_file:
        depfile = codecs.open(stp_dep, 'r', encoding='utf-8')
        inputlines = depfile.readlines()
    else:
        inputlines = stp_dep.split('\n')

    dpg_list = []
    dep_lines = []
    i = 0
    
    for line in inputlines:
        if line.strip():
            dep_lines.append(line)
        else:            
            dpg = SpanGraph.init_dep_graph(dep_lines)
            dep_lines = []
            dpg_list.append(dpg)

    if not dpg.is_empty():
        dpg_list.append(dpg)

    return dpg_list


def write_parsed_amr(parsed_amr,instances, amr_file, suffix='parsed', hand_alignments=None):
    output = open(amr_file+'.'+suffix,'w')
    for pamr, inst in zip(parsed_amr,instances):
        if inst.comment:
            output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['id','date','snt-type','annotator'])))
            output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['snt','tok'])))
            if hand_alignments:
                output.write('# ::alignments %s ::gold\n' % (hand_alignments[inst.comment['id']]))
            #output.write('# %s\n' % (' '.join(('::%s %s')%(k,v) for k,v in inst.comment.items() if k in ['alignments'])))
        else:
            output.write('# ::id %s\n'%(inst.sentID))
            output.write('# ::snt %s\n'%(inst.text))

        try:
            output.write(pamr.to_amr_string())
        except TypeError:
            import pdb
            pdb.set_trace()
        output.write('\n\n')
    output.close()


def write_span_graph(span_graph_pairs, instances, amr_file, suffix='spg'):
    output_d = open(amr_file+'.'+suffix+'.dep',  'w')
    output_p = open(amr_file+'.'+suffix+'.parsed', 'w')
    output_g = open(amr_file+'.'+suffix+'.gold', 'w')

    for i in xrange(len(instances)):
        output_d.write('# id:%s\n%s' % (instances[i].comment['id'],instances[i].printDep()))
        output_p.write('# id:%s\n%s' % (instances[i].comment['id'],span_graph_pairs[i][0].print_dep_style_graph()))
        output_g.write('# id:%s\n%s' % (instances[i].comment['id'],span_graph_pairs[i][1].print_dep_style_graph()))
        output_p.write('# eval:Unlabeled Precision:%s Recall:%s F1:%s\n' % (span_graph_pairs[i][2][0],span_graph_pairs[i][2][1],span_graph_pairs[i][2][2]))
        output_p.write('# eval:Labeled Precision:%s Recall:%s F1:%s\n' % (span_graph_pairs[i][2][3],span_graph_pairs[i][2][4],span_graph_pairs[i][2][5]))
        output_p.write('# eval:Tagging Precision:%s Recall:%s\n' % (span_graph_pairs[i][2][6],span_graph_pairs[i][2][7]))
        output_d.write('\n')
        output_p.write('\n')
        output_g.write('\n')

    output_d.close()
    output_p.close()
    output_g.close()


def build_opts(parser):
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='set up verbose level for debug')
    parser.add_argument('-b', '--begin', type=int, default=0,
                        help='specify which sentence to begin the alignment or oracle testing for debug')
    parser.add_argument('-s', '--start_step', type=int, default=0,
                        help='specify which step to begin oracle testing;for debug')
    parser.add_argument('-d', '--dev', help='development file')
    parser.add_argument('-a', '--add', help='additional training file')
    parser.add_argument('-as', '--actionset', choices=['basic'], default='basic',
                        help='choose different action set')
    parser.add_argument('-m', '--mode',
                        choices=['preprocess', 'test_gold_graph', 'align', 'userGuide',
                                     'oracleGuide', 'train', 'parse', 'eval'],
                        help="preprocess:generate pos tag, dependency tree, ner\n"
                                 " align:do alignment between AMR graph and sentence string")
    parser.add_argument('-dp', '--depparser',
                        choices=['stanford', 'stanfordConvert', 'stdconv+charniak', 'clear', 'mate', 'turbo'],
                        default='stdconv+charniak',
                        help='choose the dependency parser')
    parser.add_argument('--coref', action='store_true', help='flag to enable coreference information')
    parser.add_argument('--prop', action='store_true', help='flag to enable semantic role labeling information')
    parser.add_argument('--rne', action='store_true', help='flag to enable rich name entity')
    parser.add_argument('--verblist', action='store_true', help='flag to enable verbalization list')
    parser.add_argument('--onto', choices=['onto', 'onto+bolt', 'wsj'],
                        default='wsj', help='choose which charniak parse result trained on ontonotes')
    parser.add_argument('--model', help='specify the model file')
    parser.add_argument('--feat', help='feature template file')
    parser.add_argument('-iter', '--iterations', default=1, type=int, help='training iterations')
    parser.add_argument('amr_file', nargs='?', help='amr annotation file/input sentence file for parsing')
    parser.add_argument('--prpfmt', choices=['xml', 'plain'], default='plain', help='preprocessed file format')
    parser.add_argument('--amrfmt', choices=['sent', 'amr', 'amreval'], default='sent',
                        help='specifying the input file format')
    parser.add_argument('--smatcheval', action='store_true', help='give evaluation score using smatch')
    parser.add_argument('-e', '--eval', nargs=2, help='Error Analysis: give parsed AMR file and gold AMR file')
    parser.add_argument('--section', choices=['proxy', 'all'], default='all',
                        help='choose section of the corpus. Only works for LDC2014T12 dataset.')


def do_preproces(opt):
    """
    opt.amrfmt is expected to be `plain' and prpfmt to be `plain'

    :param opt:
    :return:
    """
    preprocess(opt.amr_file, start_corenlp=True, input_format=opt.amrfmt, prp_format=opt.prpfmt)
    print "Done preprocessing!"


def do_test_gold_graph(opt):
    """

    :param opt:
    :return:
    """
    instances = preprocess(opt.amr_file, start_corenlp=False, input_format=opt.amrfmt, prp_format=opt.prpfmt)
    gold_amr = []
    for inst in instances:
        GraphState.sent = inst.tokens
        gold_amr.append(GraphState.get_parsed_amr(inst.gold_graph))
    write_parsed_amr(gold_amr, instances, amr_file, 'abt.gold')
    print "Done output AMR!"


def do_train(opt):
    # training
    print "Parser Config:"
    print "Incorporate Coref Information: %s" % constants.FLAG_COREF
    print "Incorporate SRL Information: %s" % constants.FLAG_PROP
    print "Substitue the normal name entity tag with rich name entity tag: %s" % constants.FLAG_RNE
    print "Using verbalization list: %s" % constants.FLAG_VERB
    print "Using charniak parser trained on ontonotes: %s" % constants.FLAG_ONTO
    print "Dependency parser used: %s" % constants.FLAG_DEPPARSER

    train_instances = preprocess(opt.amr_file, start_corenlp=True, input_format=opt.amrfmt, prp_format=opt.prpfmt)
    if opt.add:
        train_instances += preprocess(opt.add, start_corenlp=True, input_format=opt.amrfmt, prp_format=opt.prpfmt)
    if opt.dev:
        dev_instances = preprocess(opt.dev, start_corenlp=True, input_format=opt.amrfmt, prp_format=opt.prpfmt)

    if opt.section != 'all':
        print "Choosing corpus section: %s" % opt.section
        tcr = constants.get_corpus_range(opt.section, 'train')
        train_instances = train_instances[tcr[0]: tcr[1]]
        if args.dev:
            dcr = constants.get_corpus_range(opt.section, 'dev')
            dev_instances = dev_instances[dcr[0]: dcr[1]]
    else:
        print "Choosing all sections!"

    feat_template = opt.feat if opt.feat else None
    model = Model(elog=experiment_log)

    parser = Parser(model=model, oracle_type=DET_T2G_ORACLE_ABT, action_type=opt.actionset,
                    verbose=opt.verbose, elog=experiment_log)

    model.setup(action_type=opt.actionset, instances=train_instances, parser=parser,
                feature_templates_file=feat_template)

    print >> experiment_log, "BEGIN TRAINING!"
    best_f_score, best_p_score, best_r_score = 0., 0., 0.
    best_model = None
    best_epoch = 1

    for epoch in xrange(1, opt.iterations + 1):
        print >> experiment_log, "shuffling training instances"
        random.shuffle(train_instances)

        print >> experiment_log, "Iteration:", epoch
        begin_updates = parser.perceptron.get_num_updates()
        parser.parse_corpus_train(train_instances)
        parser.perceptron.average_weight()

        if opt.dev:
            print >> experiment_log, "Result on develop set:"
            _, parsed_amr = parser.parse_corpus_test(dev_instances)
            parsed_suffix = opt.section + '.' + opt.model.split('.')[-1] + '.' + str(epoch) + '.parsed'
            write_parsed_amr(parsed_amr, dev_instances, opt.dev, parsed_suffix)
            if opt.smatcheval:
                gold_dataset = codecs.open(opt.dev, 'r', encoding='utf-8').read().strip().split('\n\n')
                parsed_filename = opt.dev + '.' + parsed_suffix
                parsed_dataset = codecs.open(parsed_filename, 'r', encoding='utf-8').read().strip().split('\n\n')

                scorer = SmatchScorer()
                parsed_dataset = [data for data in parsed_dataset if '# ::id' in data]
                gold_dataset = [data for data in gold_dataset if '# ::id' in data]

                for parsed_data, gold_data in zip(parsed_dataset, gold_dataset):
                    parsed_data = ' '.join([l.strip() for l in parsed_data.splitlines() if not l.startswith('#')])
                    gold_data = ' '.join([l.strip() for l in gold_data.splitlines() if not l.startswith('#')])
                    try:
                        scorer.update(gold_data, parsed_data)
                    except:
                        continue
                pscore, rscore, fscore = scorer.score()
                print >> experiment_log, "F-score: ", fscore
                if fscore > best_f_score:
                    best_model = model
                    best_epoch = epoch
                    best_f_score, best_p_score, best_r_score = fscore, pscore, rscore
                    print >> experiment_log, "New best achieved, saved!"

    if best_model is not None:
        print >> experiment_log, "Best result on iteration %d:\n Precision: %f\n Recall: %f\n F-score: %f" % (
        best_epoch, best_p_score, best_r_score, best_f_score)
        best_model.save_model(opt.model + '.m')
    print >> experiment_log, "DONE TRAINING!"


def main():
    arg_parser = argparse.ArgumentParser(description="Brandeis transition-based AMR parser 1.0")
    build_opts(arg_parser)

    args = arg_parser.parse_args()

    amr_file = args.amr_file
    instances = None
    train_instance = None
    constants.FLAG_COREF = args.coref
    constants.FLAG_PROP = args.prop
    constants.FLAG_RNE = args.rne
    constants.FLAG_VERB = args.verblist
    constants.FLAG_ONTO = args.onto
    constants.FLAG_DEPPARSER = args.depparser

    if args.mode == 'preprocess':
        # using corenlp to preprocess the sentences
        do_preproces(args)

    elif args.mode == 'test_gold_graph':
        # preprocess the JAMR aligned amr
        do_test_gold_graph(args)

    elif args.mode == 'align':
        # do alignment
        if args.input_file:
            instances = pickle.load(open(args.input_file, 'rb'))
        else:
            raise ValueError("Missing data file! specify it using --input or using preprocessing!")
        gold_instances_file = args.input_file.split('.')[0] + '_gold.p'

        print >> log, "Doing alignment..."

        if LOGGED:
            saveerr = sys.stderr
            sys.stderr = open('./log/alignment.log', 'w')

        amr_aligner = Aligner(verbose=args.verbose)
        ref_graphs = []
        begin = args.begin 
        counter = 1
        for i in range(len(instances)):
            snt = instances[i].text
            amr = instances[i].amr
            if args.verbose > 1:
                print >> log, counter
                print >> log, "Sentence:"
                print >> log, snt+'\n'
                print >> log, "AMR:"
                print >> log, amr.to_amr_string()

            alresult = amr_aligner.apply_align(snt, amr)
            ref_amr_graph = SpanGraph.init_ref_graph(amr, alresult)
            instances[i].addGoldGraph(ref_amr_graph)
            if args.verbose > 1:
                print >> log, amr_aligner.print_align_result(alresult,amr)
            counter += 1

        pickle.dump(instances,  open(gold_instances_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        if LOGGED:
            sys.stderr.close() 
            sys.stderr = saveerr
        print >> log, "Done alignment and gold graph generation."
        sys.exit()
        
    elif args.mode == 'userGuide':
        # test user guide actions
        print 'Read in training instances...'
        train_instances = preprocess(amr_file,False)

        sentID = int(raw_input("Input the sent ID:"))
        amr_parser = Parser()
        amr_parser.testUserGuide(train_instances[sentID])

        sys.exit()

    elif args.mode == 'oracleGuide':
        # test deterministic oracle
        train_instances = preprocess(amr_file, start_corenlp=False, input_format=args.amrfmt, prp_format=args.prpfmt)
        try:
            hand_alignments = load_hand_alignments(amr_file+str('.hand_aligned'))
        except IOError:
            hand_alignments = []

        start_step = args.start_step
        begin = args.begin
        amr_parser = Parser(oracle_type=DET_T2G_ORACLE_ABT,verbose=args.verbose)
        #ref_graphs = pickle.load(open('./data/ref_graph.p','rb'))
        n_correct_total = .0
        n_parsed_total = .0
        n_gold_total = .0
        pseudo_gold_amr = []
        n_correct_tag_total = .0
        n_parsed_tag_total = 0.
        n_gold_tag_total = .0

        gold_amr = []
        aligned_instances = []
        for instance in train_instances[begin:]:
            
            if hand_alignments and instance.comment['id'] not in hand_alignments: continue
            state = amr_parser.testOracleGuide(instance, start_step)
            n_correct_arc, n1, n_parsed_arc, n_gold_arc, n_correct_tag, n_parsed_tag, n_gold_tag = state.evaluate()
            if n_correct_arc != n1:
                import pdb
                pdb.set_trace()
            n_correct_total += n_correct_arc
            n_parsed_total += n_parsed_arc
            n_gold_total += n_gold_arc
            p = n_correct_arc/n_parsed_arc if n_parsed_arc else .0
            r = n_correct_arc/n_gold_arc if n_gold_arc else .0
            indicator = 'PROBLEM!' if p < 0.5 else ''
            if args.verbose > 2:
                print >> sys.stderr, "Precision: %s Recall: %s  %s\n" % (p,r,indicator)
            n_correct_tag_total += n_correct_tag
            n_parsed_tag_total += n_parsed_tag
            n_gold_tag_total += n_gold_tag
            p1 = n_correct_tag/n_parsed_tag if n_parsed_tag else .0
            r1 = n_correct_tag/n_gold_tag if n_gold_tag else .0
            if args.verbose > 2:
                print >> sys.stderr, "Tagging Precision:%s Recall:%s" % (p1, r1)

            instance.comment['alignments'] +=\
                ''.join(' %s-%s|%s' % (idx-1, idx, instance.amr.get_pid(state.A.abt_node_table[idx]))
                        for idx in state.A.abt_node_table if isinstance(idx,int))

            aligned_instances.append(instance)
            pseudo_gold_amr.append(GraphState.get_parsed_amr(state.A))
        pt = n_correct_total/n_parsed_total if n_parsed_total != .0 else .0
        rt = n_correct_total/n_gold_total if n_gold_total !=.0 else .0
        ft = 2*pt*rt/(pt+rt) if pt+rt != .0 else .0
        write_parsed_amr(pseudo_gold_amr, aligned_instances, amr_file, 'pseudo-gold', hand_alignments)
        print "Total Accuracy: %s, Recall: %s, F-1: %s" % (pt,rt,ft)

        tp = n_correct_tag_total/n_parsed_tag_total if n_parsed_tag_total != .0 else .0
        tr = n_correct_tag_total/n_gold_tag_total if n_gold_tag_total != .0 else .0
        print "Tagging Precision:%s Recall:%s" % (tp,tr)

    elif args.mode == 'train':
        do_train(args)
        
    elif args.mode == 'parse':
        # actual parsing
        test_instances = preprocess(amr_file, start_corenlp=False, input_format=args.amrfmt, prp_format=args.prpfmt)
        if args.section != 'all':
            print "Choosing corpus section: %s"%(args.section)
            tcr = constants.get_corpus_range(args.section,'test')
            test_instances = test_instances[tcr[0]:tcr[1]]
            
        #random.shuffle(test_instances)
        print >> experiment_log, "Loading model: ", args.model 
        model = Model.load_model(args.model)
        parser = Parser(model=model,oracle_type=DET_T2G_ORACLE_ABT,action_type=args.actionset,verbose=args.verbose,elog=experiment_log)
        print >> experiment_log ,"BEGIN PARSING"
        span_graph_pairs,results = parser.parse_corpus_test(test_instances)
        parsed_suffix = '%s.%s.parsed'%(args.section,args.model.split('.')[-2])
        write_parsed_amr(results,test_instances,amr_file,suffix=parsed_suffix)

        print >> experiment_log, "DONE PARSING"
        if args.smatcheval:
            smatch_path = "./smatch_2.0.2/smatch.py"
            python_path = 'python'
            options = '--pr -f'
            parsed_filename = amr_file+'.'+parsed_suffix
            command = '%s %s %s %s %s' % (python_path,smatch_path,options,parsed_filename, amr_file)
                    
            print 'Evaluation using command: ' + (command)
            print subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)

    elif args.mode == 'eval':
        '''break down error analysis'''
        # TODO: here use pickled file, replace it with parsed AMR and gold AMR
        span_graph_pairs = pickle.load(open(args.eval[0],'rb'))
        instances = pickle.load(open(args.eval[1],'rb'))
        
        amr_parser = Parser(oracle_type=DET_T2G_ORACLE_ABT,verbose=args.verbose)
        error_stat = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for spg_pair,instance in zip(span_graph_pairs,instances):
            amr_parser.errorAnalyze(spg_pair[0],spg_pair[1],instance,error_stat)

    else:
        arg_parser.print_help()


if __name__ == "__main__":
    main()

