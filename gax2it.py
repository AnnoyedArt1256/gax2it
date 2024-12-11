import argparse
import os
import libs.gba as gba
import struct
import re
import math

from libs.gax_constants import (
    mixing_rates, max_channels, min_channels, max_fx_channels
)

def dopack(data):
    packbuf = []
    lc = [[253,0,255,0,0] for j in range(64)]
    lm = [0x00 for j in range(64)]
    global ch_count
    for l in data:
        for i in range(ch_count):
            c = l[i]
            m = 0x00
            if c[0] != 253:
                m |= 0x10 if c[0] == lc[i][0] else 0x01
            if c[1] != 0:
                m |= 0x20 if c[1] == lc[i][1] else 0x02
            if c[2] != 255:
                m |= 0x40 if c[2] == lc[i][2] else 0x04
            if c[3] != 0 or c[4] != 0:
                m |= 0x80 if c[3] == lc[i][3] and c[4] == lc[i][4] else 0x08

            v = i+1
            if m != lm[i]:
                v |= 0x80
            packbuf.append(v)

            if v & 0x80:
                packbuf.append(m)
                lm[i] = m

            if m & 0x01:
                packbuf.append(c[0])
                lc[i][0] = c[0]
            if m & 0x02:
                packbuf.append(c[1])
                lc[i][1] = c[1]
            if m & 0x04:
                packbuf.append(c[2])
                lc[i][2] = c[2]
            if m & 0x08:
                packbuf.append(c[3])
                packbuf.append(c[4])
                lc[i][3] = c[3]
                lc[i][4] = c[4]

        packbuf.append(0)
    return packbuf


def get_GAX_version(rom):
	'''
	Gets the GAX Sound Engine library version string from a binary
	'''
	string_regex = rb'GAX Sound Engine v?(\d)\.(\d{1,2})([a-zA-Z-]{,4}) \(([a-zA-Z]{3} *\d{1,2} \d{4})\)'	

	result = re.search(string_regex, rom)
	if result != None:
		result = re.search(string_regex, rom)[0]

	return result

def parse_song_setting_gax1(offset, rom):

    #adapted from loveemu's GAX scanner
    #from Gaxripper v1

    if offset + 0x20 >= len(rom):
        return None

    if gax_ver_mi == 99 and gax_ver_le.lower() == "f":
        settings = struct.unpack_from('<Bx4Hxx3L', rom, offset)
        settings = {
            "num_channels": settings[0],
            "step_count": settings[1],
            "num_patterns": settings[2],
            "restart_position": settings[3],
            "master_volume": settings[4],
            "seq_data_pointer": settings[5],
            "inst_data_pointer": settings[6],
            "wave_data_pointer": settings[7]
        }
    else:
        settings = struct.unpack_from('<Bx3H3L', rom, offset)
        settings = {
            "num_channels": settings[0],
            "step_count": settings[1],
            "num_patterns": settings[2],
            "restart_position": settings[3],
            "seq_data_pointer": settings[4],
            "inst_data_pointer": settings[5],
            "wave_data_pointer": settings[6]
        }

    channel_count = settings["num_channels"]

    if channel_count == 0 or channel_count > 32:
        #songs have a minimum of 1 channel, and a max of 32 channels. 
        #FX files have a minimum and maximum of 0 channels.
        #if it does happen to find a FX song settings struct, it's just going to return None anyways.
        return None

    if settings["step_count"] > 256:
        #this area of the settings struct is reserved (set to 0)
        return None

    if settings["num_patterns"] > 256:
        #this area of the settings struct is reserved (set to 0)
        return None

    #checks if the sequence, instrument and sample data pointers are valid ROM locations, or are pointers to 32-bit aligned positions
    if not gba.is_rom_address(settings["seq_data_pointer"]) or gba.from_rom_address(settings["seq_data_pointer"]) > len(rom) or settings["seq_data_pointer"] % 4 != 0:
        return None
    if not gba.is_rom_address(settings["inst_data_pointer"]) or gba.from_rom_address(settings["inst_data_pointer"]) > len(rom) or settings["inst_data_pointer"] % 4 != 0:
        return None
    if not gba.is_rom_address(settings["wave_data_pointer"]) or gba.from_rom_address(settings["wave_data_pointer"]) > len(rom) or settings["wave_data_pointer"] % 4 != 0:
        return None

    if settings["restart_position"] > settings["num_patterns"]:
        return None

    return settings

def parse_song_setting_gax2(offset, rom):

    #adapted from loveemu's GAX scanner
    #from Gaxripper v1

    if offset + 0x20 >= len(rom):
        return None

    settings = struct.unpack_from('<Bx4Hxx3LHB', rom, offset)
    settings = {
        "num_channels": settings[0],
        "step_count": settings[1],
        "num_patterns": settings[2],
        "restart_position": settings[3],
        "master_volume": settings[4],
        "seq_data_pointer": settings[5],
        "inst_data_pointer": settings[6],
        "wave_data_pointer": settings[7],
        "mixing_rate": settings[8],
        #"fx_mixing_rate": settings[9],
        "num_fx_slots": settings[9]
    }

    channel_count = settings["num_channels"]

    if channel_count == 0 or channel_count > 32:
        #songs have a minimum of 1 channel, and a max of 32 channels. 
        #FX files have a minimum and maximum of 0 channels.
        #if it does happen to find a FX song settings struct, it's just going to return None anyways.
        return None

    """
    if (rom[offset+10]|(rom[offset+11]<<8)) != 0x00:
        #this area of the settings struct is reserved (set to 0)
        return None

    if rom[offset+27] != 0x00:
        #this area of the settings struct is reserved (set to 0)
        return None
    """

    if settings["step_count"] > 256:
        #this area of the settings struct is reserved (set to 0)
        return None

    #checks if the sequence, instrument and sample data pointers are valid ROM locations, or are pointers to 32-bit aligned positions
    if not gba.is_rom_address(settings["seq_data_pointer"]) or gba.from_rom_address(settings["seq_data_pointer"]) > len(rom) or settings["seq_data_pointer"] % 4 != 0:
        return None
    if not gba.is_rom_address(settings["inst_data_pointer"]) or gba.from_rom_address(settings["inst_data_pointer"]) > len(rom) or settings["inst_data_pointer"] % 4 != 0:
        return None
    if not gba.is_rom_address(settings["wave_data_pointer"]) or gba.from_rom_address(settings["wave_data_pointer"]) > len(rom) or settings["wave_data_pointer"] % 4 != 0:
        return None

    if settings["restart_position"] > settings["num_patterns"]:
        return None

    if settings["mixing_rate"] not in mixing_rates:
        return None

    return settings

def parse_song_setting_gax3(offset, rom):

	#adapted from loveemu's GAX scanner
	#from Gaxripper v1

	if offset + 0x20 >= len(rom):
		return None

	settings = struct.unpack_from('<Bx4Hxx3L2HB', rom, offset)
	settings = {
		"num_channels": settings[0],
		"step_count": settings[1],
		"num_patterns": settings[2],
		"restart_position": settings[3],
		"master_volume": settings[4],
		"seq_data_pointer": settings[5],
		"inst_data_pointer": settings[6],
		"wave_data_pointer": settings[7],
		"mixing_rate": settings[8],
		"fx_mixing_rate": settings[9],
		"num_fx_slots": settings[10]
	}

	channel_count = settings["num_channels"]

	if channel_count == min_channels or channel_count > max_channels:
		#songs have a minimum of 1 channel, and a max of 32 channels. 
		#FX files have a minimum and maximum of 0 channels.
		#if it does happen to find a FX song settings struct, it's just going to return None anyways.
		return None

	if rom[offset+0x1e] != 0x00:
		#this area of the settings struct is reserved (set to 0)
		return None

	#checks if the sequence, instrument and sample data pointers are valid ROM locations, or are pointers to 32-bit aligned positions
	if not gba.is_rom_address(settings["seq_data_pointer"]) or gba.from_rom_address(settings["seq_data_pointer"]) > len(rom) or settings["seq_data_pointer"] % 4 != 0:
		return None
	if not gba.is_rom_address(settings["inst_data_pointer"]) or gba.from_rom_address(settings["inst_data_pointer"]) > len(rom) or settings["inst_data_pointer"] % 4 != 0:
		return None
	if not gba.is_rom_address(settings["wave_data_pointer"]) or gba.from_rom_address(settings["wave_data_pointer"]) > len(rom) or settings["wave_data_pointer"] % 4 != 0:
		return None

	if settings["mixing_rate"] not in mixing_rates: # > 48000
		return None


	#if the end offset of the channel pointers is less or equal to the ROM size, return None.
	if offset + 0x20 + (channel_count * 4) >= len(rom):
		return None

	#check here if the addresses in this struct are ROM pointers.
	for address in struct.unpack_from("<" + "L" * channel_count, rom, offset + 0x20):
		if not gba.is_rom_address(address) or address % 4 != 0:
			return None

	return settings

def scan_ROM(rom,gax_ver):
    song_setting_list = list()
    for dword in range(0, len(rom), 4):
        if gax_ver >= 3:
            song_setting = parse_song_setting_gax3(dword, rom)
        elif gax_ver == 2: # taken from GAXM
            song_setting = parse_song_setting_gax2(dword, rom)
        else:
            song_setting = parse_song_setting_gax1(dword, rom)
        if song_setting != None:
            print(">> Song setting data found at", hex(gba.to_rom_address(dword)))
            song_setting_list.append(dword)
    return song_setting_list

def lerp(v, d):
    return v[0] * (1 - d) + v[1] * d

from libs.gax_enums import (
    step_type, perf_row_effect, step_effect
)

parser = argparse.ArgumentParser()
parser.add_argument('file_path', help="GAX .gba file")


args = parser.parse_args()
#Path and file name of the file
gax_path = os.path.realpath(args.file_path)
file_name = os.path.basename(gax_path)


note_names = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']

def semitone_to_note(semitone,transp):
    if semitone == step_type.note_off:
        return "OFF"
    semitone += transp
    return note_names[(semitone+11) % 12] + str(max(min(math.floor((semitone+11)/12),9),0))

f = 0

def fprintf(f,w):
    f.write(w.encode("ascii"))

def write8(w):
    f.write(bytearray([int(w)&0xff]))


def write16(w):
    f.write(bytearray([int(w)&0xff,int(w)>>8&0xff]))

def write32(w):
    f.write(bytearray([int(w)&0xff,int(w)>>8&0xff,int(w)>>16&0xff,int(w)>>24&0xff]))

def conv_int8(n):
    if n >= 0x80:
        return n-0x100
    else:
        return n


def dump_step_data(step_cmd,transp):

    def dumpEffect(step_cmd):

        effect_str = ''
        unknown_effect = False

        if type(step_cmd.effect_type) == gax.step_effect:

            if step_cmd.effect_type == gax.step_effect.set_volume:
                effect_str = '{:0>2}'.format(f'{step_cmd.effect_param:X}')
                return effect_str
            else:
                effect_str = '{:0>2}'.format(f'{step_cmd.effect_type.value:X}')

        if type(step_cmd.effect_type) == int:
            effect_str = '{:0>2}'.format(f'{step_cmd.effect_type:X}')
            unknown_effect = True

        elif step_cmd.effect_type == None:
            effect_str = '..'

        if step_cmd.effect_param != None:
            effect_str += '{:0>2}'.format(f'{step_cmd.effect_param:X}')
        else:
            effect_str += '..'

        return effect_str

    #we just uncompressed the rle data

    if step_cmd == None: # empty step
        return "...........|"

    elif step_cmd.semitone != None and semitone_to_note(step_cmd.semitone,transp) == "OFF": # note off
        effect_str = dumpEffect(step_cmd)
        if step_cmd.effect_type == gax.step_effect.set_volume:
            return '===..' + effect_str + '....|'
        else:
            return "===...." + effect_str + "|"
    else:

        if step_cmd.semitone == None: # effect only
            effect_str = dumpEffect(step_cmd)

            if step_cmd.effect_type == gax.step_effect.set_volume:
                return '.....' + effect_str + '....|'
            else:
                return "......." + effect_str + "|"

        if step_cmd.effect_type == None and step_cmd.effect_param == None: #note only
            return semitone_to_note(step_cmd.semitone,transp) + '{:0>2}'.format(f'{step_cmd.instrument:X}') + "FF....|"

        effect_str = dumpEffect(step_cmd)
        if step_cmd.instrument == 0:
            if type(step_cmd.effect_type) == gax.step_effect:
                if step_cmd.effect_type == gax.step_effect.set_volume:
                    return semitone_to_note(step_cmd.semitone,transp) + ".." + effect_str + '....|'
            if effect_str[:2] == "03":
                return semitone_to_note(step_cmd.semitone,transp) + '....' + effect_str + '|'
            return semitone_to_note(step_cmd.semitone,transp) + '..FF' + effect_str + '|'
        else:
            # note + effect
            if step_cmd.effect_type == gax.step_effect.set_volume:
                return semitone_to_note(step_cmd.semitone,transp) + '{:0>2}'.format(f'{step_cmd.instrument:X}') + effect_str + '....|'
            else:
                return semitone_to_note(step_cmd.semitone,transp) + '{:0>2}'.format(f'{step_cmd.instrument:X}') + 'FF' + effect_str + '|'


with open(gax_path, "rb") as f:
    gba_rom = f.read()
    #detect the GAX version used in the ROM
    gax_library = get_GAX_version(gba_rom).decode('iso-8859-1')

    print("> GAX library version |", gax_library)
    print("> Scanning for valid song settings..\n")

    string_regex = r'GAX Sound Engine v?(\d)\.(\d{1,2})([a-zA-Z-]{,4})'
    
    result = re.match(string_regex, gax_library)
    gax_ver = int(result.groups()[0])
    gax_ver_mi = int(result.groups()[1])
    gax_ver_le = str(result.groups()[2])

    song_settings = scan_ROM(gba_rom,gax_ver)
    print("\n Songs found |", str(len(song_settings)))

    if gax_ver >= 3:
        import libs.shinen_gax3 as gax
    elif gax_ver == 2 or (gax_ver_mi == 99 and gax_ver_le.lower() == "f"):
        import libs.shinen_gax2 as gax
    else:
        import libs.shinen_gax1 as gax




for song_id in range(len(song_settings)):
    gax_obj = gax.unpackGAXFromROM([song_settings[song_id]], gba_rom)

    print("\n> Composed by", gax_obj.get_auth())
    
    song_count = gax_obj.get_song_count()

    print("> Song", '{:0>2}'.format(f'{song_id}') + ":", gax_obj.get_song_name(0) + '\n')

    song_obj = gax_obj.get_song_data(0)
    #print(song_obj.dump_order_list())

    ch_count = song_obj.get_properties().channel_count+1
    num_pat = song_obj.get_properties().song_length
    ins_amt = len(gax_obj.instrument_set)
    pat_rows = song_obj.get_properties().step_count

    song_name = str(gax_obj.get_song_name(0))
    if song_name == "":
        song_name = "_%s"%str(hex(song_settings[song_id])[2:])
    file_name = "./out/%s.it"%str(song_name)

    idx = 0
    pre_samples = []
    ins_lut = []
    loop_pos = []
    rate_pos = []
    for waveform in gax_obj.wave_set.wave_bank:
        if len(waveform) > 0:
            if not gax_ver >= 3:
                waveform = [x+128&0xff for x in waveform]
            pre_samples.append(waveform)
            ins_lut.append(idx)
            idx += 1
        else:
            rate_pos.append(0)
            loop_pos.append(0)
            ins_lut.append(0)

    samples = []
    arps = []
    for ins in range(ins_amt):
        index = gax_obj.instrument_set[ins].perf_list['perf_list_data']
        arps.append([])
        n = 0
        for i in index:
            if i['note'] != 0:
                n = conv_int8(i['note'])

            if i['fixed']:
                p = (conv_int8(i['note'])+14)|(1<<30)
            else:
                p = n
            arps[ins].append(p)

        loop_start = -1
        loop_end = -1
        ping_pong = False
        finetune = 0
        index = gax_obj.instrument_set[ins].perf_list['perf_list_data']
        index = [x['wave_slot_id'] for x in index if x['wave_slot_id'] != 0]
        if len(index) > 0:
            index = index[-1]
        else:
            index = 0
        if index > 0:
            idx = 0
            for i in range(4):
                 if i == (index-1):
                      break
                 if gax_obj.instrument_set[ins].header['wave_slots'][i] != 0:
                      idx += 1
            w = gax_obj.instrument_set[ins].wave_params[idx]
            loop_start = w['loop_start']
            loop_end = w['loop_end']
            ping_pong = w['ping_pong']
            finetune = w['finetune']
            index = gax_obj.instrument_set[ins].header['wave_slots'][index-1]
        waveform = list(pre_samples[ins_lut[index]])

        if index > 0:
            if loop_start == loop_end and loop_start == 0: loop_start, loop_end = (None,None)
            loop_start = 0xFFFFFFFF if loop_start == None else loop_start
            loop_end = 0xFFFFFFFF if loop_end == None else loop_end

            if w['start_position'] < min(len(waveform),loop_start,loop_end) and not w['modulate']:
                waveform = waveform[w['start_position']:]
                if loop_start != 0xFFFFFFFF:
                    loop_start -= w['start_position']
                if loop_end != 0xFFFFFFFF:
                    loop_end -= w['start_position']

            if loop_end != 0xFFFFFFFF:
                waveform = waveform[:loop_end]

            if w['modulate']:
                speed = max(16//max(w['modulate_speed'],1),1)
                loop_size = loop_end-loop_start
                loop_len = math.floor(loop_size/w['modulate_step'])
                waveform_loop = waveform[loop_start:loop_end]
                waveform = waveform[:loop_start]
                for l in range(0,loop_len):
                    start = l*w['modulate_step']
                    waveform.extend(waveform_loop[start:(start+w['modulate_size'])]*speed)
                for l in range(loop_len-1,0,-1):
                    start = l*w['modulate_step']
                    waveform.extend(waveform_loop[start:(start+w['modulate_size'])]*speed)
                loop_end = len(waveform)
            elif ping_pong:
                waveform += reversed(waveform[loop_start:loop_end])
                loop_end = len(waveform)

        real_samp = int(33452*(2**(float((finetune/32)-56+17)/12.0)))
        #real_samp = int(15769*(2**(float((finetune/32)-57+30)/12.0)))
        l = len(waveform)

        if l > 400000:
            del waveform[400000:]
            l = 400000
            loop_start = min(loop_start,400000)
            loop_end = min(loop_end,400000)

        samp_rate = 15769

        if index > 0:
            samp_rate = real_samp

        s = [samp_rate,loop_start,loop_end,waveform]

        samples.append(s)

    f = open(file_name,"wb")

    fprintf(f,"IMPM")

    for i in range(26): write8(0)

    write8(4)
    write8(16)

    write16(num_pat)
    write16(ins_amt)
    write16(len(samples))
    write16(num_pat)

    write16(0x201)
    write16(0x201)

    write16((1<<2)|(1<<3))
    write16(0)

    write8(127) # global vol
    write8(127) # mixing vol

    write8(6)
    write8(150)

    write8(63)
    write8(0)

    write16(0)
    write32(0)

    write32(0)

    for i in range(64): write8(32)
    for i in range(64): write8(63) # channel vol

    for i in range(num_pat): write8(i)

    ins_pointer_pos = f.tell()
    for i in range(4*ins_amt): write8(0)

    samp_pointer_pos = f.tell()
    for i in range(4*len(samples)): write8(0)

    pattern_pos = f.tell()
    for i in range(4*num_pat): write8(0)

    #write16(0) # edit history (?)

    for ins in range(ins_amt):
        index = gax_obj.instrument_set[ins].perf_list['perf_list_data']

        vol_mul = []
        vol = 255
        idx = 0
        for i in range(256):
            for j in index[idx]['effect']:
                if j[1] == perf_row_effect.set_volume:
                    vol = j[0]
                elif j[1] == perf_row_effect.jump_to_row:
                    idx = j[0]-1
                elif j[1] != perf_row_effect.no_effect:
                    pass
                    #print(ins,j[1])
            vol_mul.append(vol)
            idx += 1
            if idx >= len(index):
                break

        volenv = gax_obj.instrument_set[ins].volume_envelope
        points = volenv['points']

        insfile = []

        insfile.extend("IMPI".encode("ascii"))
        insfile.extend([0]*12)
        insfile.append(0)

        insfile.append(0) # NNA action (?)
        insfile.append(0)
        insfile.append(0)

        insfile.append(0)
        insfile.append(0)

        insfile.append(0)
        insfile.append(0)

        insfile.append(128)
        insfile.append(1<<7)
        insfile.append(0)
        insfile.append(0)

        insfile.extend([0]*4) # midi bank (?)

        str_ins = "ins"+str(ins)
        for t in range(26):
            if t < len(str_ins):
                insfile.append(ord(str_ins[t]))
            else:
                insfile.append(0)

        insfile.append(127) # filter cutoff (?)
        insfile.append(15) # filter res (?)

        insfile.extend([0]*4) # midi bank (?)

        a = int(arps[ins][-1])
        for i in range(120):
            if (a&int(1<<30)) > 0:
                insfile.append(max(min((a&0xff)+11,255),0))
                insfile.append(ins+1)
            else:
                insfile.append(max(min(i+a+12,255),0))
                insfile.append(ins+1)

        bitfield = 1<<2
        bitfield |= 1<<1 if volenv['loop_start'] != None and volenv['loop_end'] != None else 0
        bitfield |= 1
        insfile.append(bitfield)
        l = min(len(points),21)
        insfile.append(l)

        if volenv['sustain_point'] == None and volenv['loop_start'] != None and volenv['loop_end'] != None:
            insfile.append(volenv['sustain_point'] if volenv['sustain_point'] != None else l+1)
            insfile.append(volenv['sustain_point'] if volenv['sustain_point'] != None else l+1)

            insfile.append(volenv['loop_start'] if volenv['loop_start'] != None else 0)
            insfile.append(volenv['loop_end'] if volenv['loop_end'] != None else 0)
        else:
            insfile.append(volenv['loop_start'] if volenv['loop_start'] != None else 0)
            insfile.append(volenv['loop_end'] if volenv['loop_end'] != None else 0)
    
            insfile.append(volenv['sustain_point'] if volenv['sustain_point'] != None else l+1)
            insfile.append(volenv['sustain_point'] if volenv['sustain_point'] != None else l+1)

        v = 0

        for i in range(25):
            if i < min(len(points),20):
                insfile.append(int(points[i][1]*vol_mul[min(v,len(vol_mul)-1)]/255)>>2)
                insfile.append(v>>0&0xff)
                insfile.append(v>>8&0xff)
                if i < (len(points)-1): v += abs(points[i+1][0]-points[i][0])
                else: v += 1
            else:
                insfile.append(0)
                insfile.append(v>>0&0xff)
                insfile.append(v>>8&0xff)

        insfile.append(0)
        insfile.append(1)
        insfile.append(2)
        insfile.append(0)
        insfile.append(0)
        insfile.append(0)
        insfile.append(0)
        insfile.extend([0,0,0,0,1,0])
        insfile.extend([0,0,0]*23)


        insfile.append(0)
        insfile.append(1)
        insfile.append(2)
        insfile.append(0)
        insfile.append(0)
        insfile.append(0)
        insfile.append(0)
        insfile.extend([0,0,0,0,1,0])
        insfile.extend([0,0,0]*23)
        insfile.extend([0]*7)

        f.seek(0,2)
        POS = f.tell()
        f.write(bytearray(insfile))
        f.seek(ins_pointer_pos+(ins*4))
        write32(POS)

    for s in range(len(samples)):
        f.seek(0,2)
        POS = f.tell()


        samp_rate, loop_start, loop_end, samp = samples[s]
        has_loop = loop_start != 0xFFFFFFFF and loop_end != 0xFFFFFFFF

        insfile = []

        insfile.extend("IMPS".encode("ascii"))
        insfile.extend([0]*12)
        insfile.append(0)
        insfile.append(64)
        insfile.append(has_loop<<4|1)
        insfile.append(64)

        str_ins = "sample"+str(s)
        for t in range(26):
            if t < len(str_ins):
                insfile.append(ord(str_ins[t]))
            else:
                insfile.append(0)

        insfile.append(0)
        insfile.append(0)
        l = len(samp)
        insfile.extend([l&0xff,l>>8&0xff,l>>16&0xff,l>>24&0xff])
        l = loop_start
        insfile.extend([l&0xff,l>>8&0xff,l>>16&0xff,l>>24&0xff])
        l = loop_end
        insfile.extend([l&0xff,l>>8&0xff,l>>16&0xff,l>>24&0xff])
        l = samp_rate
        insfile.extend([l&0xff,l>>8&0xff,l>>16&0xff,l>>24&0xff])
        insfile.extend([0]*8)
        l = POS+len(insfile)+8
        insfile.extend([l&0xff,l>>8&0xff,l>>16&0xff,l>>24&0xff])

        vibrato_params = gax_obj.instrument_set[s].header['vibrato_params']

        depth = int(max(min(vibrato_params["vibrato_depth"],255),0))
        tick = vibrato_params["vibrato_wait"]
        if tick == 0:
            delay_div = 0
        else:
            delay_div = (depth*256)//tick
        if vibrato_params["vibrato_speed"] == 0:
            speed = 0
        else:
            speed = min(4*vibrato_params["vibrato_speed"],255)
        delay_div = int(min(max(delay_div,0),255))
        insfile.extend([speed,depth,delay_div,0])

        insfile.extend(samp)

        f.write(bytearray(insfile))
        f.seek(samp_pointer_pos+(s*4))
        write32(POS)

    f.seek(0,2)

    did_slide = [0]*64
    notes = [[0,0]]*64
    speeds = [0,0]
    real_speeds = [6,6]

    wrote_0Bxx = False
    c = -1
    row_ind = 0
    old_speed = 10000

    ins_val = [0]*64
    perf_pos = [-1]*64
    perf_speed = [1]*64
    perf_speed_pos = [0]*64
    perf_eff = [[0xFFFF,0xFFFF]]*64
    another_eff = [[0xFFFF,0xFFFF]]*64

    for i in range(num_pat): 
        s = f.tell()
        write16(0)
        write16(pat_rows)
        write32(0)

        c += 1

        row_len = song_obj.get_properties().step_count
        data = [[[253,0,255,0,0] for j in range(64)] for i in range(pat_rows)]
        for a in range(64):
            if a < song_obj.get_properties().channel_count:
                for b in range(song_obj.get_properties().step_count):
                    pattern_id = song_obj.get_order_list()[a][c][0]
                    transp = song_obj.get_order_list()[a][c][1]
                    pat_buffer = [0x0000,0x0000,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF] 
    
                    if song_obj.patterns[pattern_id] != None:
                        p = dump_step_data(song_obj.patterns[pattern_id][b],transp)
                        #print(p)
                        if p[:3] == "===":
                            pat_buffer[0] = 101 #100
                            notes[a] = [0,0]
                        elif p[:3] != "...":
                            semi = ["C-","C#","D-","D#","E-","F-","F#","G-","G#","A-","A#","B-"]
                            note = semi.index(p[:2])                        
                            pat_buffer[0] = note+1
                            pat_buffer[1] = int(p[2])
                            notes[a] = pat_buffer[:2]
                            another_eff[a] = [0xFFFF,0xFFFF]
                        if p[3:5] != ".." and p[3:5] != "00":
                            pat_buffer[2] = int(p[3:5],16)
                            ins_val[a] = pat_buffer[2]
                        if p[5:7] != "..":
                            pat_buffer[3] = int(p[5:7],16)
                        if p[7:9] != "..":
                            #eff = int(p[7:9],16)
                            eff = [0,1,2,3,0,0xB,0,7,0,0,0xF8,0xF9,0,0x0D,0xED,0x0F][int(p[7:9],16)]
                            if eff != 0:
                                pat_buffer[4] = eff                        
                                if p[9:11] != "..":
                                    pat_buffer[5] = int(p[9:11],16)
                                if eff == 1 or eff == 2:
                                    if pat_buffer[5] > 1:
                                            pat_buffer[5] = min(max(pat_buffer[5]>>1,0),255)
                                if eff == 3:
                                    if pat_buffer[5] != 0:
                                            pat_buffer[5] = int(min(max((0x80-((pat_buffer[5]-1)*4))*1.5,0),255))
                                    another_eff[a] = [0x03,pat_buffer[5]]
                                else:
                                    another_eff[a] = [0xFFFF,0xFFFF]

                        if p[:3] != "..." and p[:3] != "===" and pat_buffer[2] == 0xFFFF and pat_buffer[4] != 0x03:
                            pat_buffer[4] = 0x03
                            pat_buffer[5] = 0xFF

                        if b == (pat_rows-1) and c == (num_pat-1) and pat_buffer[4] == 0xFFFF and not wrote_0Bxx:
                            pat_buffer[4] = 0x0B
                            pat_buffer[5] = song_obj.get_properties().restart_position
                            wrote_0Bxx = True

                    old_transp = song_obj.get_order_list()[a][c-1][1]
                    if b == 0 and notes[a] != [0,0] and pat_buffer[:2] == [0,0] and old_transp != transp:
                        note = (notes[a][0]-1)+(notes[a][1]*12)
                        note = (note-old_transp)+transp
                        note = max(note,0)
                        notes[a] = [(note%12)+1,note//12]
                        pat_buffer[:2] = notes[a]
                        notes[a] = [0,0]

                    if pat_buffer[0] == 101:
                        perf_pos[a] = -1
                        perf_eff[a] = [0xFFFF,0xFFFF]
                    elif pat_buffer[0] != 0:
                        perf_pos[a] = 0
                        perf_eff[a] = [0xFFFF,0xFFFF]
                        perf_speed_pos[a] = 0
                        perf_speed[a] = 1

                    if perf_pos[a] >= 0:
                        index = gax_obj.instrument_set[ins_val[a]].perf_list['perf_list_data']
                        #if ins_val[a] == 53 or ins_val[a] == 54: print(index)
                        for speed in range(real_speeds[b%2]):
                            if perf_pos[a] == -1: break
                            perf_speed_pos[a] += 1
                            if perf_speed_pos[a] >= perf_speed[a]:
                                perf_speed_pos[a] = 0
                                for j in index[perf_pos[a]]['effect']:
                                    if j[1] == perf_row_effect.set_volume:
                                       pass
                                    elif j[1] == perf_row_effect.jump_to_row:
                                        perf_pos[a] = j[0]-1
                                    elif j[1] == perf_row_effect.no_effect:
                                        pass
                                    elif j[1] == perf_row_effect.pitch_slide_up:
                                        perf_eff[a] = [0x01,min(int(j[0]/1.5),255)|1024]
                                    elif j[1] == perf_row_effect.pitch_slide_down:
                                        perf_eff[a] = [0x02,min(int(j[0]/1.5),255)|1024]
                                    elif j[1] == perf_row_effect.set_speed:
                                        perf_speed[a] = j[0]
                                    else:
                                        pass #print(j)
                                perf_pos[a] += 1
                                if perf_pos[a] >= len(index):
                                    perf_pos[a] = -1
                                    break

                        if (pat_buffer[4:6] == [0xFFFF,0xFFFF] or pat_buffer[4:6] == [0x03,0xFF]) and perf_eff[a] != [0xFFFF,0xFFFF]:
                            pat_buffer[4:6] = perf_eff[a]

                    if (pat_buffer[4:6] == [0xFFFF,0xFFFF] or pat_buffer[4:6] == [0x03,0xFF]) and another_eff[a] != [0xFFFF,0xFFFF]:
                            pat_buffer[4:6] = another_eff[a]

                    if pat_buffer[0] == 101:
                        data[b][a][0] = 0xFF
                    elif pat_buffer[0] != 0:
                        data[b][a][0] = (pat_buffer[0]-1)+(pat_buffer[1]*12)

                    if pat_buffer[2] != 0xFFFF:
                        data[b][a][1] = pat_buffer[2]+1
                    if pat_buffer[3] != 0xFFFF:
                        data[b][a][2] = pat_buffer[3]>>2

                    if pat_buffer[4] != 0xFFFF:
                        eff = pat_buffer[4]
                        if eff == 2:
                            data[b][a][3] = ord("E")
                            if pat_buffer[5]&1024:
                                pat_buffer[5] &= 255
                                if pat_buffer[5] < 4:
                                    data[b][a][4] = min(pat_buffer[5]<<2,15)&0xf|0xf0
                                elif pat_buffer[5] == 0:
                                    data[b][a][4] = 0
                                else:
                                    data[b][a][4] = max(pat_buffer[5]>>3,1)
                            else:
                                if pat_buffer[5] < 4:
                                    data[b][a][4] = pat_buffer[5]
                                else:
                                    data[b][a][4] = pat_buffer[5]>>2
                        elif eff == 1:
                            data[b][a][3] = ord("F")
                            if pat_buffer[5]&1024:
                                pat_buffer[5] &= 255
                                if pat_buffer[5] < 4:
                                    data[b][a][4] = min(pat_buffer[5]<<2,15)&0xf|0xf0
                                elif pat_buffer[5] == 0:
                                    data[b][a][4] = 0
                                else:
                                    data[b][a][4] = max(pat_buffer[5]>>3,1)
                            else:
                                if pat_buffer[5] < 4:
                                    data[b][a][4] = pat_buffer[5]
                                else:
                                    data[b][a][4] = pat_buffer[5]>>2
                        elif eff == 3:
                            data[b][a][3] = ord("G")
                            data[b][a][4] = pat_buffer[5]
                        elif eff == 0xF:
                            data[b][a][3] = ord("A")
                            data[b][a][4] = pat_buffer[5]
                            speeds = [0,0]
                            real_speeds = [pat_buffer[5],pat_buffer[5]]
                        elif eff == 0xB:
                            data[b][a][3] = ord("B")
                            data[b][a][4] = pat_buffer[5]
                            row_len = b+1
                        elif eff == 0xD:
                            data[b][a][3] = ord("C")
                            data[b][a][4] = pat_buffer[5]
                            row_len = b+1
                        elif eff == 0xED:
                            data[b][a][3] = ord("S")
                            data[b][a][4] = 0xD0|(min(pat_buffer[5],15))
                        elif eff == 0xF9:
                            data[b][a][3] = ord("D")
                            if pat_buffer[5] < 8:
                                data[b][a][4] = min(pat_buffer[5]<<1,15)&0xf|0xf0
                            else:
                                data[b][a][4] = min(pat_buffer[5]>>2,15)&0xf
                        elif eff == 0xF8:
                            data[b][a][3] = ord("D")
                            if pat_buffer[5] < 8:
                                data[b][a][4] = min(pat_buffer[5]<<1,15)<<4|0x0f
                            else:
                                data[b][a][4] = min(pat_buffer[5]>>2,15)<<4|0x0f
                        elif eff == 7:
                            speeds = [pat_buffer[5]>>4&0xf,pat_buffer[5]>>0&0xf]
                            real_speeds = [pat_buffer[5]>>4&0xf,pat_buffer[5]>>0&0xf]
                        if data[b][a][3] >= ord("A"):
                            data[b][a][3] -= ord("A")-1
            if a == (ch_count-1):
                for b in range(row_len):
                    if speeds[row_ind%2] != old_speed and speeds != [0,0]:
                        data[b][a][3] = ord("A")
                        data[b][a][4] = speeds[row_ind%2]
                        if data[b][a][3] >= ord("A"):
                            data[b][a][3] -= ord("A")-1
                    row_ind += 1

        f.write(bytearray(dopack(data)))

        s2 = f.tell()
        f.seek(s)
        write16(s2-s-8)
        f.seek(pattern_pos+(i*4))
        write32(s)
        f.seek(0,2)
    print(hex(song_settings[song_id]))
    print(song_obj.properties.mixing_rate)
