import struct, math, itertools

def parse_fmr(path):
    with open(path, 'rb') as f:
        data = f.read()
    size_x = struct.unpack('>H', data[14:16])[0]
    size_y = struct.unpack('>H', data[16:18])[0]
    num_views = data[22]
    offset = 24
    views = []
    for v in range(num_views):
        num_minutiae = data[offset+3]
        offset += 4
        minutiae = []
        for m in range(num_minutiae):
            chunk = data[offset:offset+6]
            x_raw, y_raw, angle, mquality = struct.unpack('>HHBB', chunk)
            mtype = (x_raw >> 14) & 0x03
            x = x_raw & 0x3FFF
            y = y_raw & 0x3FFF
            angle_deg = angle * (360.0/256.0)
            minutiae.append({'x': x, 'y': y, 'type': mtype, 'angle': angle_deg})
            offset += 6
        views.append(minutiae)
    return {'size_x': size_x, 'size_y': size_y, 'views': views}


def relative_tokens(minutiae, k=2, dist_bin=10, dir_bin=20, orient_bin=20, include_type=True):
    """
    For each minutia, look at its k nearest neighbours and encode the
    relationship as (distance_bin, relative_direction_bin, delta_orientation_bin).
    This is invariant to global translation/rotation because everything is
    expressed relative to minutia i's own position and orientation.
    """
    n = len(minutiae)
    tokens = set()

    for i in range(n):
        xi, yi, ai = minutiae[i]['x'], minutiae[i]['y'], minutiae[i]['angle']

        # find k nearest neighbours of i
        dists = []
        for j in range(n):
            if i == j:
                continue
            xj, yj = minutiae[j]['x'], minutiae[j]['y']
            d = math.hypot(xj - xi, yj - yi)
            dists.append((d, j))
        dists.sort(key=lambda t: t[0])
        neighbours = dists[:k]

        for d, j in neighbours:
            xj, yj, aj = minutiae[j]['x'], minutiae[j]['y'], minutiae[j]['angle']

            # direction from i to j, relative to i's own ridge orientation
            abs_dir = math.degrees(math.atan2(yj - yi, xj - xi)) % 360
            rel_dir = (abs_dir - ai) % 360

            # difference in ridge orientation between i and j
            delta_orient = (aj - ai) % 360

            db = int(d // dist_bin)
            rb = int(rel_dir // dir_bin)
            ob = int(delta_orient // orient_bin)

            if include_type:
                token = (db, rb, ob, minutiae[i]['type'], minutiae[j]['type'])
            else:
                token = (db, rb, ob)
            tokens.add(token)

    return tokens


if __name__ == '__main__':
    paths = ['/mnt/user-data/uploads/1_1.ist', '/mnt/user-data/uploads/1_2.ist', '/mnt/user-data/uploads/1_3.ist']
    for k in [3, 4, 6]:
        print(f'--- k={k} nearest neighbours ---')
        sets = {}
        for path in paths:
            r = parse_fmr(path)
            sets[path] = relative_tokens(r['views'][0], k=k)
        for a, b in itertools.combinations(sets.keys(), 2):
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            print(f'  {a.split("/")[-1]} vs {b.split("/")[-1]}: |A|={len(sets[a])} |B|={len(sets[b])} jaccard={inter/union:.3f}')
        print()