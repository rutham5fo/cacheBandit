import os
import copy
import re
import pyslang
from jinja2 import Environment, FileSystemLoader

class modNode:

    def __init__(self, name, fdata, file_extension, parent=None, level=0):
        self.name = name                # Module name
        self.fdata = fdata              # Module definition Datafile
        self.fext = file_extension
        self.parent = parent            # Root node has no parent (None)
        self.children = []              # Children of this module
        self.pending = []               # Pending signals raised by children for processing. Proccesed in the order received
                                        # {'modName': <module_name>, 'netName': <debug net name>, 'op': <operation performed>}
        self.level = level              # The nodes absolute level in the hierarchy; Root at 0
        # Search related
        self.actv_pat = {               # Aactivation patterns for all search patterns. Activation patterns must be hit for search pattern to be matched
            'outport': '(^|\\s)(module)([#(]|\\s)',
            'assign': ''
        }
        self.search_pat = {             # All patterns to search for in a file on hook_setup are placed in this dict as: {pat_name: search_string}
            'outport': '.*' + re.escape(');'),
            'assign': '(^|\\s)(assign)(\\s)'
        }
        self.search_res = {}            # The result of a search during hook_setup run are placed in this dict as: {pat_name, loc in search_hooks list}
        self.search_hooks = []          # The line loc of hooks stored in the order of discovery
    
    def set_search_pattern(self, pattern):
        self.search_pat.update(pattern)
    
    def push_pending(self, signals):
        for x in signals:
            self.pending.append(x)
    
    def get_pending(self):
        return self.pending
    
    def add_children(self, children):
        for child_name in children:
            self.children.append(child_name)
            actv_pat = f'(^|\\s)({child_name})([#(]|\\s)'
            search_pat = '.*' + re.escape(');')
            self.actv_pat.update({child_name: actv_pat})
            self.search_pat.update({child_name: search_pat})
        #print(f'{modNode.add_children.__name__} ||| Updated children = {self.children}, actv_pat = {self.actv_pat}, search_pat = {self.search_pat}')

    def get_fdata(self):
        # Return a copy
        fd = copy.deepcopy(self.fdata)
        return fd
    
    def write_fdata(self, dest):
        fname = os.path.join(dest, f'{self.name}.{self.fext}')
        try:
            with open(fname, 'w') as fo:
                fo.writelines(self.fdata)
        except Exception as e:
            print(f'{modNode.write_fdata.__name__} ||| WARNING: File {fname} could not be written')
        
    def find_hooks(self):
        pat_actv = dict.fromkeys(self.actv_pat.keys(), 0)
        for fid, fd in enumerate(self.fdata):
            #print(f'\n{modNode.find_hooks.__name__} ||| line[{fid}] = {fd[:-1]}', flush=True)
            for k in self.actv_pat.keys():
                actv_pat = self.actv_pat[k]
                amatch = re.search(actv_pat, fd)
                #print(f'{modNode.find_hooks.__name__} ||| actv_pat = {actv_pat}, match = {amatch}')
                if (amatch):
                    pat_actv.update({k: 1})
                    #print(f'{modNode.find_hooks.__name__} ||| [{k}] pat_actv = {pat_actv}', flush=True)
            for k in self.search_pat.keys():
                search_pat = self.search_pat[k]
                pmatch = re.search(search_pat, fd)
                #print(f'{modNode.find_hooks.__name__} ||| search_pat = {search_pat}, match = {pmatch}')
                if (pmatch and pat_actv[k] == 1):
                    #print(f'{modNode.find_hooks.__name__} ||| Hook found for [{k}] pattern[{search_pat}] @ line[{fid}]', flush=True)
                    # Check if match is already in the list
                    if k not in self.search_res.keys():
                        # Add match to result
                        cur_loc = len(self.search_hooks)
                        self.search_res[k] = cur_loc
                        # Add resulting line to hook
                        self.search_hooks.append(fid)
                    else:
                        cur_loc = self.search_res[k]
                        # Update aboslute loc in hooks list
                        self.search_hooks[cur_loc] = fid
                    #print(f'{modNode.find_hooks.__name__} ||| search_res = {self.search_res} | search_hooks = {self.search_hooks}', flush=True)
                    # Deactivate pattern
                    pat_actv[k] = 0
                    break
        #print(f'{modNode.find_hooks.__name__} ||| Hooks_result = {self.search_res}, Hooks_loc (un-processed) = {self.search_hooks}', flush=True)
        # Process hooks and store the relative differences
        thooks = copy.deepcopy(self.search_hooks)
        for hid, h in enumerate(thooks):
            if hid == 0: continue
            self.search_hooks[hid] = h - thooks[hid-1]
        #print(f'{modNode.find_hooks.__name__} ||| Hooks_result = {self.search_res}, Hooks_loc = {self.search_hooks}')

    def get_hook(self, hook_name):
        loc = self.search_res[hook_name]
        acc = 0
        for h in range(loc):
            acc += self.search_hooks[h]
        fd_loc = acc + self.search_hooks[loc]
        return fd_loc

    def update_hook(self, hook_name, val):
        loc = self.search_res[hook_name]
        self.search_hooks[loc] += val
        #print(f'{modNode.update_hook.__name__} ||| Updated hooks: search_hooks = {self.search_hooks}, search_res = {self.search_res}')

    def outport_template(self, name, width):
        return f'\t\toutput wire [{width-1}:0]\t\t\t\t{name}'
    
    def assign_template(self, lhs_name, rhs_name):
        return f'\tassign {lhs_name} = {rhs_name};'

    def pin_template(self, pin_name, net_name):
        return f'\t\t.{pin_name}({net_name})'
    
    def net_raise(self, active_level=0):
        if (active_level >= self.level and self.children):
            # Raise all pending nets from children
            objs_raise = []
            still_pending = []
            # Create job bins
            #print(f'{modNode.net_raise.__name__} ||| Node children = {self.children}')
            port_jobs = {k: [] for k in self.children}
            child_jobs = {k: [] for k in self.children}
            #print(f'{modNode.net_raise.__name__} ||| Pending jobs = {self.pending}')
            #print(f'{modNode.net_raise.__name__} ||| port_jobs = {port_jobs}, child_jobs = {child_jobs}')
            for oid, obj in enumerate(self.pending):
                if (obj['op'] == 'raise'):
                    child_name = obj['modName']
                    child_width = int(obj['width'])
                    child_net = obj['netName']
                    port_line = self.outport_template(child_net, child_width) + '\n'
                    child_line = self.pin_template(child_net, child_net) + '\n'
                    port_jobs[child_name].append(port_line)
                    child_jobs[child_name].append(child_line)
                    tobj = copy.deepcopy(obj)
                    tobj['modName'] = self.name
                    objs_raise.append(tobj)
                else:
                    # Add cur job to still pending list when not done
                    still_pending.append(obj)
            #print(f'{modNode.net_raise.__name__} ||| port_jobs = {port_jobs}, child_jobs = {child_jobs}')
            # Complete jobs from each child
            for ch in self.children:
                pj = port_jobs[ch]
                cj = child_jobs[ch]
                self.adder(pj, 'outport')
                self.adder(cj, ch)
            # Remove completed jobs
            self.pending = still_pending
            #print(f'{modNode.net_raise.__name__} ||| Pending jobs = {self.pending}')
            # Raise new jobs
            #print(f'{modNode.net_raise.__name__} ||| New jobs raised = {objs_raise}')
            return objs_raise

    def net_parse(self, modName, dbg_nets, line, lid):
        # Set *KEEP* attribute to all debug nets in HDL
        rep_line = line
        rep_out = None
        rep_assign = None
        pat_ignore = r'^\s*//'
        hit_id = None
        print(f'{modNode.net_parse.__name__} ||| Parsing line[{lid}] = {line}')
        for nid, dbg in enumerate(dbg_nets):
            nmod = dbg['modName']
            nnet = dbg['netName']
            ntype = dbg['netType']
            nattr = dbg['attribute']
            nwidth = int(dbg['width'])
            typePat = f'(^|\\s)({ntype})'
            rep_template = f'{nattr}'
            netPat = f'(?>{nnet})\\s*;'
            # Control set
            ignore = re.search(pat_ignore, line)    # Ignore all lines commented out
            nhit = re.search(typePat, line)          # Hit all lines containing wires/logic
            nvld = re.search(netPat, line)      # Pick only the line which contains key(net_name) from all hits
            #print(f'{modNode.net_parse.__name__} ||| current line[{lid}], hit={nhit}, ignore={ignore}, vld={nvld}: {line}')
            if nmod == modName and nhit and nvld and not ignore:
                # Generate Line placement
                rep_pad = nhit.group(1) # Extract the leading whitespaces
                rep_with = rep_pad + rep_template + nhit.group(2)        # Construct replacement string
                rep_line = re.sub(typePat, rep_with, line)          # Replace the line
                hit_id = nid
                # Generate output port
                if (nwidth != 0):
                    port_name = f'dbg_{self.name}_{nnet}'
                    rep_out = self.outport_template(port_name, nwidth)
                    rep_assign = self.assign_template(port_name, nnet)
                #print(f'{modNode.net_parse.__name__} ||| NetName[{hit_id}] = {nnet}, Replaced line = {rep_line}')
                #print(f'{modNode.net_parse.__name__} ||| nwidth = {nwidth}, rep_out = {rep_out}, rep_assign = {rep_assign}')
                break
        # The parser returns: [dbg_net_id, Line post replacement, generated output port, generated assign]
        return hit_id, rep_line, rep_out, rep_assign

    def adder(self, new_line=None, group=None):
        if (new_line is not None and group is not None):
            base_loc = self.get_hook(group)
            if (group == 'assign'):
                base_loc = self.get_hook(group) + 1
            else:
                base_loc = self.get_hook(group)
            #print(f'{modNode.adder.__name__} ||| Adder job = {new_line}, len = {len(new_line)} | group = {group} | start loc = {base_loc}', flush=True)
            for nid, n in enumerate(new_line):
                nloc = base_loc + nid
                if (group != 'assign'):
                    # Add ',' to prev line if its not empty
                    prev_line = self.fdata[nloc-1][:-1]         # Leave the newline char
                    ignore_line = re.search(r'^\s*//', prev_line)
                    chars_present = re.search(r'\S', prev_line)
                    if (chars_present and not ignore_line):
                        # Remove comment
                        prev_line = re.sub('//.*$', "", prev_line)
                        #print(f'{modNode.adder.__name__} ||| Finding comment = {re.search(r'\\')}')
                        # Remove trailing whitspace
                        prev_line = prev_line.rstrip()
                        self.fdata[nloc-1] = prev_line + ',\n'
                #print(f'{modNode.adder.__name__} ||| Adder inserting @[{nloc}] line = {n[:-1]}', flush=True)
                self.fdata.insert(nloc, n)
            # Update group locs by sending number of lines added to this group
            self.update_hook(group, len(new_line))

    def feeder(self, metadata):
        """
        :param list metadata: List of one or more nets, which must be processed as a batch
        :param str parser: Parser name passed for callback from here
        """
        nets_done = []
        add_ports = []
        add_assigns = []
        push_pending = []
        # Call parser on metadata for all lines
        #print(f'{modNode.feeder.__name__} ||| Feeder: metadata = {metadata}')
        for fid, fd in enumerate(self.fdata):
            # Parser returns: [meta_id, new_line, op (add), group (corresponding hook name to add at)]
            meta_id, replace_line, port_line, assign_line = self.net_parse(self.name, metadata, fd[:-1], fid)
            #print(f'{modNode.feeder.__name__} ||| replace_line = {replace_line}, meta_id = {meta_id}, port_line = {port_line}, assign_line = {assign_line}')
            # Replace line
            #print(f'{modNode.feeder.__name__} ||| replaced = {replace_line}, net_id = {meta_id}')
            self.fdata[fid] = replace_line + '\n'
            if (meta_id is not None):
                nets_done.append(meta_id)
            if (assign_line is not None):
                tline = assign_line + '\n'
                add_assigns.append(tline)
            if (port_line is not None):
                tline = port_line + '\n'
                add_ports.append(tline)
                pending_template = {'modName': self.name, 'netName': f'dbg_{self.name}_{metadata[meta_id]["netName"]}', 'width': f'{metadata[meta_id]["width"]}', 'op': 'raise'}
                push_pending.append(pending_template)
                #print(f'{modNode.feeder.__name__} ||| pushing to parent[{self.parent}], pending = {pending_template}')
        # Insert lines
        self.adder(add_ports, 'outport')
        self.adder(add_assigns, 'assign')
        # Remove done nets
        new_md = []
        for mid, md in enumerate(metadata):
            if (mid not in nets_done):
                new_md.append(md)
        # Return nets that are not done
        #print(f'{modNode.feeder.__name__} ||| Nets_pending = {new_md}')
        #print(f'{modNode.feeder.__name__} ||| Pushed pending to parent = {push_pending}')
        return new_md, push_pending

class modTree:

    def __init__(self, top_module=None, hdl_dir=None, compile_unit='default'):
        """
        :param str compile_unit: valid values -> default | verilog | systemverilog. Types of files to search while building module tree
        """
        self.compile_unit = compile_unit
        self.hdl_dir = hdl_dir
        self.root = top_module
        self.tree = {}                  # {'<modName>': <modNode_object>, ..., }
        self.tree_depth = 0             # Total depth of the tree
        self.hdl_lib = []               # Subset of files related to top_module (root) from the hdl_dir (source)
        self.discovery_order = []       # Order of linear traversal from root to leaves and vice-versa (BFS)
        self.discover_nodes = []        # List of modules, yet to be discovered (BFS ends when list is empty) | (child_name, level, parent)
        if (self.root is not None and self.hdl_dir is not None):
            self.build_tree(self.root)
    
    def tree_reset(self):
        self.root = None
        self.hdl_dir = None
        self.hdl_lib = []
        self.tree_depth = 0
        self.discovery_order = []
        self.discover_nodes = []
        self.tree = dict.fromkeys(self.tree, 0)

    def set_top(self, top_module):
        self.root = top_module
    
    def set_hdl_dir(self, hdl_dir):
        self.hdl_dir = hdl_dir

    def get_top(self):
        return self.root
    
    def get_hdl_dir(self):
        return self.hdl_dir

    def add_node(self, name, node):
        self.discovery_order.append(name)
        self.tree[name] = node
    
    def get_node(self, name):
        return self.tree[name]
    
    def print_tree(self):
        cur_lvl = 0
        print(f'\n{modTree.print_tree.__name__} ||| Tree Level[{cur_lvl}]:')
        for node_name in self.discovery_order:
            node_lvl = self.tree[node_name].level
            if (node_lvl > cur_lvl):
                print(f'\n{modTree.print_tree.__name__} ||| Tree Level[{node_lvl}]:')
                cur_lvl = node_lvl
            print(f'{modTree.print_tree.__name__} ||| module = {node_name}')

    def build_tree(self, top, parent=None, level=0, init=True):
        if (init):
            # Read All source files to find root self.hdl_dir.file
            file_pat = ''
            if (self.compile_unit == 'verilog'): file_pat = '.*([.]v)$'
            elif (self.compile_unit == 'systemverilog'): file_pat = '.*([.]sv)$'
            else: file_pat = '.*[.sv]$'
            files = os.listdir(self.hdl_dir)
            #print(f'{modTree.build_tree.__name__} ||| init = {init}, top_module = {top}, hdl_dir = {self.hdl_dir}, file_pat = {file_pat} | source_files = {files}')
            #_ = [print(f'{modTree.build_tree.__name__} ||| match = {re.match(file_pat, f).group(0) if (re.match(file_pat, f)) else None}') for f in files]
            self.hdl_lib = [re.match(file_pat, f).group(0) for f in files if (os.path.isfile(os.path.join(self.hdl_dir, f)) and re.match(file_pat, f))]
            #print(f'{modTree.build_tree.__name__} ||| Library built from @ src[{self.hdl_dir}]: {self.hdl_lib}')
            init = False
            self.discover_nodes = []
            self.discovery_order = []
            level = 0           # Force level to 0 on init
        else:
            # Remove current node from queue
            self.discover_nodes.pop(0)
        # Get top module's metadata
        #print(f'{modTree.build_tree.__name__} ||| Top = {top} | lib = {self.hdl_lib}')
        fdata = []
        fname_pat = f'(^{top})(.*[.sv]$)'
        #_ = [print(f'{modTree.build_tree.__name__} ||| Level[{level}] || fname = {f}, pattern = {fname_pat}, match = {re.match(fname_pat, f).group(0) if (re.match(fname_pat, f)) else None}') for f in self.hdl_lib]
        fname = [os.path.join(self.hdl_dir, re.match(fname_pat, f).group(0)) for f in self.hdl_lib if (re.match(fname_pat, f))]
        #print(f'{modTree.build_tree.__name__} ||| Top file_name = {fname[0]}')
        try:
            # Parse file to extract submodules (children) using pyslang
            ftree = pyslang.SyntaxTree.fromFile(fname[0])
            fmod = ftree.root.members[0]
            submods = [item for item in fmod.members if (re.search('.*HierarchyInstantiation.*', str(item.kind)))]
            # Read file
            with open(fname[0]) as fi:
                fdata = fi.readlines()
        except IndexError as e:
            print(f'{modTree.build_tree.__name__} ||| {e} || WARNING: File not found for Module = {top} @ level[{level}] | Handle this manually')
            return
        # Create node for module
        file_ext = 'v' if (self.compile_unit == 'verilog') else 'sv'
        node = modNode(top, fdata, file_ext, parent, level)
        # Insert node into tree
        self.add_node(top, node)
        # Get children
        children = [(s[1].valueText, level+1, top) for s in submods]
        # Add children
        child_name = [ch[0] for ch in children]
        #print(f'{modTree.build_tree.__name__} ||| Adding children = {children}')
        node.add_children(child_name)
        self.discover_nodes = self.discover_nodes + children if (children) else self.discover_nodes           # Used for calling/monitoring recursive runs
        next_node, next_level, next_parent = self.discover_nodes[0]
        self.tree_depth = level
        if (self.discover_nodes):
            self.build_tree(next_node, parent=next_parent, level=next_level, init=init)
    
    def set_hooks(self):
        for name in self.tree.keys():
            node = self.tree[name]
            node.find_hooks()
    
    def write(self, hdl_dest):
        for node in self.tree.values():
            node.write_fdata(hdl_dest)

# Utilities
def generate_from_template(top, tmpl_include, tmpl_fname, tmpl_context, tmpl_dest):
    #write_fname = out_fname
    jin = Environment(loader=FileSystemLoader(tmpl_include), trim_blocks=True, lstrip_blocks=True)
    template = jin.get_template(tmpl_fname)
    context = {'ctxt': tmpl_context}
    #print(f'{generate_from_template.__name__} ||| context = {context} \n')
    # Get template file's extension
    text = re.sub('(.*[.])', "", tmpl_fname)
    #print(f'{generate_from_template.__name__} ||| Template extension = {text}')
    write_fname = os.path.join(tmpl_dest, f'{top}_run.{text}')
    with open(write_fname, mode="w", encoding="utf-8") as genfile:
        genfile.write(template.render(context))
        #print(f'{generate_from_template.__name__} ||| ... wrote {write_fname} \n')

def get_sdc(top, src, exten='xdc'):
    # Read All source files to find root src.file
    file_pat = '.*.(sdc|xdc)$'
    files = os.listdir(src)
    #_ = [print(f'{get_sdc.__name__} ||| match = {re.match(file_pat, f).group(0)}') for f in files]
    lib = [re.match(file_pat, f).group(0) for f in files if (os.path.isfile(os.path.join(src, f)) and re.match(file_pat, f))]
    #print(f'{get_sdc.__name__} ||| All files @ src[{src}]: {lib}')
    # Get top module's constraints
    fdata = {}
    fname_pat = f'(^{top})(.*.{exten}$)'
    #fname_pat = f'(^{top})(.*.(sdc|xdc)$)'
    #_ = [print(f'{get_sdc.__name__} ||| fname = {f}, pattern = {fname_pat}, match = {re.match(fname_pat, f).group(0) if (re.match(fname_pat, f)) else None}') for f in lib]
    constraints = [re.match(fname_pat, f).group(0) for f in lib if (re.match(fname_pat, f))]
    for fname in constraints:
        fpath = os.path.join(src, fname)
        with open(fpath) as fi:
            fdata[fname] = fi.readlines()
    return fdata

def put_sdc(constraints, dest):
    # Write output file(s)
    fname = constraints.keys()
    for fn in fname:
        dest_file = os.path.join(dest, fn)
        fdata = constraints[fn]
        with open(dest_file, 'w') as fo:
            fo.writelines(fdata)

def debug_net_pass(design_tree, dbg_nets):
    # Add KEEP attribute to HDL
    # Call KEEP parser for each debug net, with its corresponding module feeder
    #check_dbg_nets = copy.deepcopy(dbg_nets)
    for node in design_tree.tree.values():
        parent_name = node.parent
        #print(f'{debug_net_pass.__name__} ||| delim_beg_pat = {delim_pattern_beg}')
        dbg_nets, push_pending = node.feeder(dbg_nets)
        if (push_pending and parent_name is not None):
            # Prepare node list for next pass
            push_node = design_tree.get_node(node.parent)
            push_node.push_pending(push_pending)
            #print(f'{debug_net_pass.__name__} ||| Node[{node.name}] pushing to parent[{push_node.name}]: {push_pending}')

def debug_raise_pass(design_tree):
    # Run raise pass for N steps where N is the design tree depth
    N = design_tree.tree_depth
    for lp in range(N):
        actv_lvl = N-lp
        for node in design_tree.tree.values():
            push_pending = node.net_raise(actv_lvl)
            parent_name = node.parent
            # push jobs into parent
            if (push_pending and parent_name is not None):
                push_node = design_tree.tree[parent_name]
                push_node.push_pending(push_pending)
                #print(f'{debug_raise_pass.__name__} ||| Node[{node.name}] pushing to parent[{push_node.name}]: {push_pending}')
    #_ = [print(f'{debug_raise_pass.__name__} ||| Node[{node.name}] pending = {len(node.pending)}') for node in design_tree.tree.values()]

def gen_sdc(top, src, dest, extn='xdc'):
    fdata = get_sdc(top, src, extn)
    put_sdc(fdata, dest)

def gen_script(top, part, run_name, synth_en, place_en, route_en, bitstream_en, tmpl_include, dbg_ports, dbg_params, tmpl_dest, tmpl_name, wsl_mode=0):
    # Create template context
    tmpl_ctxt = {}
    tmpl_ctxt['top'] = top
    tmpl_ctxt['part'] = part
    tmpl_ctxt['run_name'] = run_name
    tmpl_ctxt['synth_en'] = 1 if (synth_en) else 0
    tmpl_ctxt['place_en'] = 1 if (place_en) else 0
    tmpl_ctxt['route_en'] = 1 if (route_en) else 0
    tmpl_ctxt['bitstream_en'] = 1 if (bitstream_en) else 0
    tmpl_ctxt['dbg_ports'] = dbg_ports
    tmpl_ctxt['dbg_params'] = dbg_params
    # Generate template
    generate_from_template(top, tmpl_include, tmpl_name, tmpl_ctxt, tmpl_dest)

def edit_make(file, params):
    # Build EXTRA_ARGS; ignore CLK_Q and CLK_UNIT param
    param_list = [f'-G{pname}={pvalue}' for pname, pvalue in params.items() if (pname != 'CLK_Q' and pname != 'CLK_UNIT')]
    args = " ".join(param_list)
    extra_args = 'EXTRA_ARGS += ' + args + '\n'
    # Open makefile
    fd = None
    with open(file) as fi:
        fd = fi.readlines()
    # Check for 'EXTRA_ARGS'
    rid = None
    if (fd):
        for lid, line in enumerate(fd):
            # Find EXTRA_ARGS line
            #print(f'{edit_make.__name__} ||| Line[{lid}] = {line} | match = {re.search('(.*EXTRA_ARGS.*)', line)}')
            if (re.search('(.*EXTRA_ARGS.*)', line)):
                rid = lid
                break
        if (rid is not None):
            #print(f'{edit_make.__name__} ||| Replacing line[{rid}] with "{extra_args}"')
            fd[rid] = extra_args
        else:
            #print(f'{edit_make.__name__} ||| Inserting at 0 "{extra_args}"')
            fd.insert(0, extra_args)
        # Writeback makefile
        with open(file, 'w') as fo:
            fo.writelines(fd)
