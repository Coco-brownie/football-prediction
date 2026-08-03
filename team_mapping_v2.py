# team_mapping_v2.py
# 双层结构：{联赛配置键: {原始名称(英文/中文双兼容): (标准英文名, 中文名称)}}
# 自动兜底、空值容错、适配数据库 D1/EPL/LLA/SER/LIG
# 2026-07-23 优化：空值过滤、去重、日志提示、入参容错、重复键清理
# 2026-07-24 09:55 修正：删除西甲错误桑德兰、清理重复冗余键
# 2026-07-24 12:00 重大修复：关闭跨联赛匹配；补齐五大联赛全部缺失球队
# 2026-07-24 19:30 兼容修复：数据库球队为中文，所有联赛追加中文队名key，中英文双匹配
def _norm_team_name(s):
    """内部辅助：归一化队名（变音/前缀/空格）"""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # 1. 变音映射
    accent_map = {
        "á": "a", "à": "a", "ä": "a", "â": "a", "ā": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e", "ē": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "ô": "o", "ö": "o", "ō": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u", "ū": "u",
        "ç": "c", "ñ": "n", "ß": "ss",
        "Á": "A", "À": "A", "Ä": "A", "Â": "A",
        "É": "E", "È": "E", "Ê": "E", "Ë": "E",
        "Í": "I", "Ì": "I", "Î": "I", "Ï": "I",
        "Ó": "O", "Ò": "O", "Ô": "O", "Ö": "O",
        "Ú": "U", "Ù": "U", "Û": "U", "Ü": "U",
        "Ç": "C", "Ñ": "N",
    }
    s = "".join(accent_map.get(c, c) for c in s)

    # 2. 统一常见前缀
    prefixes = [
        "1. FC", "FC", "CF", "CD", "RCD", "SSC",
        "AC", "AS", "US", "SV", "AFC", "GFC", "GF38"
    ]
    for p in prefixes:
        if s.startswith(p + " "):
            s = s[len(p) + 1:]
            break

    # 3. 去多余空格、转小写
    s = " ".join(s.split()).lower()
    return s

LEAGUE_TEAM_MAP = {
    # ==================== 英超 EPL 完整全量（中英文双key兼容） ====================
    "EPL": {
        "Man Utd": ("Manchester United", "曼联"),
        "Man United": ("Manchester United", "曼联"),
        "Manchester United": ("Manchester United", "曼联"),
        "曼联": ("Manchester United", "曼联"),

        "Man City": ("Manchester City", "曼城"),
        "Manchester City": ("Manchester City", "曼城"),
        "曼城": ("Manchester City", "曼城"),

        "Liverpool": ("Liverpool", "利物浦"),
        "利物浦": ("Liverpool", "利物浦"),

        "Arsenal": ("Arsenal", "阿森纳"),
        "阿森纳": ("Arsenal", "阿森纳"),

        "Chelsea": ("Chelsea", "切尔西"),
        "切尔西": ("Chelsea", "切尔西"),

        "Tottenham": ("Tottenham Hotspur", "热刺"),
        "Tottenham Hotspur": ("Tottenham Hotspur", "热刺"),
        "热刺": ("Tottenham Hotspur", "热刺"),

        "Newcastle": ("Newcastle United", "纽卡斯尔联"),
        "Newcastle United": ("Newcastle United", "纽卡斯尔联"),
        "纽卡斯尔联": ("Newcastle United", "纽卡斯尔联"),

        "Brighton": ("Brighton & Hove Albion", "布莱顿"),
        "Brighton & Hove Albion": ("Brighton & Hove Albion", "布莱顿"),
        "布莱顿": ("Brighton & Hove Albion", "布莱顿"),

        "Crystal Palace": ("Crystal Palace", "水晶宫"),
        "水晶宫": ("Crystal Palace", "水晶宫"),

        "Brentford": ("Brentford", "布伦特福德"),
        "布伦特福德": ("Brentford", "布伦特福德"),

        "Everton": ("Everton", "埃弗顿"),
        "埃弗顿": ("Everton", "埃弗顿"),

        "Ipswich Town": ("Ipswich Town", "伊普斯维奇"),
        "Ipswich": ("Ipswich Town", "伊普斯维奇"),
        "伊普斯维奇": ("Ipswich Town", "伊普斯维奇"),

        "Southampton": ("Southampton", "南安普敦"),
        "南安普敦": ("Southampton", "南安普敦"),

        "Aston Villa": ("Aston Villa", "阿斯顿维拉"),
        "阿斯顿维拉": ("Aston Villa", "阿斯顿维拉"),

        "Fulham": ("Fulham", "富勒姆"),
        "富勒姆": ("Fulham", "富勒姆"),

        "Leeds": ("Leeds United", "利兹联"),
        "Leeds United": ("Leeds United", "利兹联"),
        "利兹联": ("Leeds United", "利兹联"),

        "Leicester": ("Leicester City", "莱斯特城"),
        "Leicester City": ("Leicester City", "莱斯特城"),
        "莱斯特城": ("Leicester City", "莱斯特城"),

        "Nott'm Forest": ("Nottingham Forest", "诺丁汉森林"),
        "Nottingham Forest": ("Nottingham Forest", "诺丁汉森林"),
        "诺丁汉森林": ("Nottingham Forest", "诺丁汉森林"),

        "Sheffield United": ("Sheffield United", "谢菲尔德联"),
        "谢菲尔德联": ("Sheffield United", "谢菲尔德联"),

        "West Brom": ("West Bromwich Albion", "西布朗"),
        "West Bromwich Albion": ("West Bromwich Albion", "西布朗"),
        "西布朗": ("West Bromwich Albion", "西布朗"),

        "West Ham": ("West Ham United", "西汉姆联"),
        "West Ham United": ("West Ham United", "西汉姆联"),
        "西汉姆联": ("West Ham United", "西汉姆联"),

        "Wolves": ("Wolverhampton Wanderers", "狼队"),
        "Wolverhampton Wanderers": ("Wolverhampton Wanderers", "狼队"),
        "狼队": ("Wolverhampton Wanderers", "狼队"),

        "Bournemouth": ("AFC Bournemouth", "伯恩茅斯"),
        "AFC Bournemouth": ("AFC Bournemouth", "伯恩茅斯"),
        "伯恩茅斯": ("AFC Bournemouth", "伯恩茅斯"),

        "Burnley": ("Burnley", "伯恩利"),
        "伯恩利": ("Burnley", "伯恩利"),

        "Cardiff": ("Cardiff City", "加的夫城"),
        "Cardiff City": ("Cardiff City", "加的夫城"),
        "加的夫城": ("Cardiff City", "加的夫城"),

        "Huddersfield": ("Huddersfield Town", "哈德斯菲尔德"),
        "Huddersfield Town": ("Huddersfield Town", "哈德斯菲尔德"),
        "哈德斯菲尔德": ("Huddersfield Town", "哈德斯菲尔德"),

        "Luton": ("Luton Town", "卢顿"),
        "Luton Town": ("Luton Town", "卢顿"),
        "卢顿": ("Luton Town", "卢顿"),

        "Norwich": ("Norwich City", "诺维奇"),
        "Norwich City": ("Norwich City", "诺维奇"),
        "诺维奇": ("Norwich City", "诺维奇"),

        "Stoke": ("Stoke City", "斯托克城"),
        "Stoke City": ("Stoke City", "斯托克城"),
        "斯托克城": ("Stoke City", "斯托克城"),

        "Swansea": ("Swansea City", "斯旺西"),
        "Swansea City": ("Swansea City", "斯旺西"),
        "斯旺西": ("Swansea City", "斯旺西"),

        "Watford": ("Watford", "沃特福德"),
        "沃特福德": ("Watford", "沃特福德"),

        "Sunderland": ("Sunderland", "桑德兰"),
        "桑德兰": ("Sunderland", "桑德兰"),

        "Wigan": ("Wigan Athletic", "维冈竞技"),
        "Wigan Athletic": ("Wigan Athletic", "维冈竞技"),
        "维冈竞技": ("Wigan Athletic", "维冈竞技"),

        # 2025-26赛季升班马
        "Ipswich": ("Ipswich Town", "伊普斯维奇"),
        "Ipswich Town": ("Ipswich Town", "伊普斯维奇"),
        "伊普斯维奇": ("Ipswich Town", "伊普斯维奇"),

        "Coventry": ("Coventry City", "考文垂"),
        "Coventry City": ("Coventry City", "考文垂"),
        "考文垂": ("Coventry City", "考文垂"),

        "Hull City": ("Hull City", "赫尔城"),
        "赫尔城": ("Hull City", "赫尔城"),

        "Birmingham": ("Birmingham City", "伯明翰"),
        "伯明翰": ("Birmingham City", "伯明翰"),

        "Blackburn": ("Blackburn Rovers", "布莱克本"),
        "布莱克本": ("Blackburn Rovers", "布莱克本"),
        "Blackpool": ("Blackpool", "布莱克浦"),
        "布莱克浦": ("Blackpool", "布莱克浦"),

        "Bolton": ("Bolton Wanderers", "博尔顿"),
        "博尔顿": ("Bolton Wanderers", "博尔顿"),

        "Hull": ("Hull City", "赫尔城"),
        "赫尔城": ("Hull City", "赫尔城"),

        "Middlesbrough": ("Middlesbrough", "米德尔斯堡"),
        "米德尔斯堡": ("Middlesbrough", "米德尔斯堡"),

        "Portsmouth": ("Portsmouth", "朴茨茅斯"),
        "朴茨茅斯": ("Portsmouth", "朴茨茅斯"),

        "QPR": ("Queens Park Rangers", "女王公园巡游者"),
        "女王公园巡游者": ("Queens Park Rangers", "女王公园巡游者"),

        "Reading": ("Reading", "雷丁"),
        "雷丁": ("Reading", "雷丁"),

        "Wigan": ("Wigan Athletic", "维冈竞技"),
        "维冈竞技": ("Wigan Athletic", "维冈竞技"),

        "Birmingham City": ("Birmingham City", "伯明翰"),
        "Blackburn Rovers": ("Blackburn Rovers", "布莱克本"),
        "Blackpool": ("Blackpool", "布莱克浦"),
        "Bolton Wanderers": ("Bolton Wanderers", "博尔顿"),
        "Portsmouth": ("Portsmouth", "朴茨茅斯"),
        "Queens Park Rangers": ("Queens Park Rangers", "女王公园巡游者"),
        "Queen's Park Rangers": ("Queens Park Rangers", "女王公园巡游者"),
    },

    # ==================== 西甲 LLA 完整全量（中英文双key兼容） ====================
    "LLA": {
        "Barcelona": ("Barcelona", "巴塞罗那"),
        "Barca": ("Barcelona", "巴塞罗那"),
        "巴塞罗那": ("Barcelona", "巴塞罗那"),

        "Real Madrid": ("Real Madrid", "皇家马德里"),
        "皇家马德里": ("Real Madrid", "皇家马德里"),

        "Valencia": ("Valencia", "瓦伦西亚"),
        "瓦伦西亚": ("Valencia", "瓦伦西亚"),

        "Sevilla": ("Sevilla", "塞维利亚"),
        "塞维利亚": ("Sevilla", "塞维利亚"),

        "Girona": ("Girona", "赫罗纳"),
        "赫罗纳": ("Girona", "赫罗纳"),

        "Alaves": ("Alavés", "阿拉维斯"),
        "阿拉维斯": ("Alavés", "阿拉维斯"),

        "Almeria": ("Almería", "阿尔梅里亚"),
        "阿尔梅里亚": ("Almería", "阿尔梅里亚"),

        "Ath Bilbao": ("Athletic Bilbao", "毕尔巴鄂竞技"),
        "毕尔巴鄂竞技": ("Athletic Bilbao", "毕尔巴鄂竞技"),

        "Ath Madrid": ("Atlético Madrid", "马德里竞技"),
        "马德里竞技": ("Atlético Madrid", "马德里竞技"),

        "Cadiz": ("Cádiz", "加的斯"),
        "加的斯": ("Cádiz", "加的斯"),

        "Eibar": ("Eibar", "埃瓦尔"),
        "埃瓦尔": ("Eibar", "埃瓦尔"),

        "Elche": ("Elche", "埃尔切"),
        "埃尔切": ("Elche", "埃尔切"),

        "Espanyol": ("Espanyol", "西班牙人"),
        "Espanol": ("RCD Espanyol", "西班牙人"),
        "西班牙人": ("Espanyol", "西班牙人"),

        "Getafe": ("Getafe", "赫塔费"),
        "赫塔费": ("Getafe", "赫塔费"),

        "Granada": ("Granada", "格拉纳达"),
        "格拉纳达": ("Granada", "格拉纳达"),

        "Huesca": ("Huesca", "韦斯卡"),
        "韦斯卡": ("Huesca", "韦斯卡"),

        "La Coruna": ("Deportivo La Coruña", "拉科鲁尼亚"),
        "拉科鲁尼亚": ("Deportivo La Coruña", "拉科鲁尼亚"),

        "Las Palmas": ("Las Palmas", "拉斯帕尔马斯"),
        "拉斯帕尔马斯": ("Las Palmas", "拉斯帕尔马斯"),

        "Leganes": ("Leganés", "莱加内斯"),
        "莱加内斯": ("Leganés", "莱加内斯"),

        "Mallorca": ("Mallorca", "马略卡"),
        "马略卡": ("Mallorca", "马略卡"),

        "Osasuna": ("Osasuna", "奥萨苏纳"),
        "奥萨苏纳": ("Osasuna", "奥萨苏纳"),

        "Rayo Vallecano": ("Rayo Vallecano", "巴列卡诺"),
        "Vallecano": ("Rayo Vallecano", "巴列卡诺"),
        "Vallecas": ("Rayo Vallecano", "巴列卡诺"),
        "巴列卡诺": ("Rayo Vallecano", "巴列卡诺"),

        "Sociedad": ("Real Sociedad", "皇家社会"),
        "皇家社会": ("Real Sociedad", "皇家社会"),

        "Valladolid": ("Valladolid", "巴利亚多利德"),
        "巴利亚多利德": ("Valladolid", "巴利亚多利德"),

        "Villarreal": ("Villarreal", "比利亚雷亚尔"),
        "比利亚雷亚尔": ("Villarreal", "比利亚雷亚尔"),

        "Betis": ("Real Betis", "皇家贝蒂斯"),
        "皇家贝蒂斯": ("Real Betis", "皇家贝蒂斯"),

        "Celta": ("Celta Vigo", "塞尔塔"),
        "塞尔塔": ("Celta Vigo", "塞尔塔"),

        "Levante": ("Levante", "莱万特"),
        "莱万特": ("Levante", "莱万特"),

        "Malaga": ("Málaga", "马拉加"),
        "马拉加": ("Málaga", "马拉加"),

        "Oviedo": ("Real Oviedo", "奥维耶多"),
        "奥维耶多": ("Real Oviedo", "奥维耶多"),

        "Cordoba": ("Córdoba CF", "科尔多瓦"),
        "科尔多瓦": ("Córdoba CF", "科尔多瓦"),

        "Hercules": ("Hércules CF", "赫库斯"),
        "赫库斯": ("Hércules CF", "赫库斯"),

        "Santander": ("Racing Santander", "桑坦德竞技"),
        "桑坦德竞技": ("Racing Santander", "桑坦德竞技"),

        "Sp Gijon": ("Sporting Gijón", "希洪竞技"),
        "希洪竞技": ("Sporting Gijón", "希洪竞技"),

        "Tenerife": ("CD Tenerife", "特内里费"),
        "特内里费": ("CD Tenerife", "特内里费"),

        "Xerez": ("Xerez CD", "赫雷斯"),
        "赫雷斯": ("Xerez CD", "赫雷斯"),

        "Zaragoza": ("Real Zaragoza", "萨拉戈萨"),
        "萨拉戈萨": ("Real Zaragoza", "萨拉戈萨"),

        # ===== 赛程CSV完整名称兼容补充 =====
        "Atletico Madrid": ("Atlético Madrid", "马德里竞技"),
        "Atlético Madrid": ("Atlético Madrid", "马德里竞技"),
        "Celta Vigo": ("Celta Vigo", "塞尔塔"),
        "Deportivo La Coruna": ("Deportivo La Coruña", "拉科鲁尼亚"),
        "Real Betis": ("Real Betis", "皇家贝蒂斯"),
        "Athletic Club": ("Athletic Bilbao", "毕尔巴鄂竞技"),
        "Athletic Bilbao": ("Athletic Bilbao", "毕尔巴鄂竞技"),
        "Real Sociedad": ("Real Sociedad", "皇家社会"),
        "Racing Santander": ("Racing Santander", "桑坦德竞技"),
        "桑坦德竞技": ("Racing Santander", "桑坦德竞技"),
        "Atlético Madrid": ("Atlético Madrid", "马德里竞技"),
        "Cádiz CF": ("Cádiz", "加的斯"),
        "Córdoba CF": ("Córdoba CF", "科尔多瓦"),
        "Deportivo La Coruña": ("Deportivo La Coruña", "拉科鲁尼亚"),
        "Hércules CF": ("Hércules CF", "赫库斯"),
        "RCD Espanyol": ("RCD Espanyol", "西班牙人"),
        "Racing Santander": ("Racing Santander", "桑坦德竞技"),
        "Sporting Gijón": ("Sporting Gijón", "希洪竞技"),
        "Xerez CD": ("Xerez CD", "赫雷斯"),
        "CD Tenerife": ("CD Tenerife", "特内里费"),
        "Leganés": ("Leganés", "莱加内斯"),
        "Málaga CF": ("Málaga", "马拉加"),
        "Real Oviedo": ("Real Oviedo", "奥维耶多"),
        "Real Zaragoza": ("Real Zaragoza", "萨拉戈萨"),

    },

    # ==================== 德甲 BUN（数据库编码D1，中英文双key兼容） ====================
    "BUN": {
        "FC Koln": ("FC Koln", "科隆"),
        "科隆": ("FC Koln", "科隆"),

        "Hoffenheim": ("Hoffenheim", "霍芬海姆"),
        "霍芬海姆": ("Hoffenheim", "霍芬海姆"),

        "Leverkusen": ("Bayer Leverkusen", "勒沃库森"),
        "勒沃库森": ("Bayer Leverkusen", "勒沃库森"),

        "Dortmund": ("Borussia Dortmund", "多特蒙德"),
        "多特蒙德": ("Borussia Dortmund", "多特蒙德"),

        "Freiburg": ("SC Freiburg", "弗赖堡"),
        "弗赖堡": ("SC Freiburg", "弗赖堡"),

        "Hertha": ("Hertha Berlin", "柏林赫塔"),
        "柏林赫塔": ("Hertha Berlin", "柏林赫塔"),

        "Mainz": ("Mainz 05", "美因茨"),
        "美因茨": ("Mainz 05", "美因茨"),

        "M'gladbach": ("Borussia Mönchengladbach", "门兴格拉德巴赫"),
        "门兴格拉德巴赫": ("Borussia Mönchengladbach", "门兴格拉德巴赫"),

        "Schalke 04": ("Schalke 04", "沙尔克04"),
        "沙尔克04": ("Schalke 04", "沙尔克04"),

        "Werder Bremen": ("Werder Bremen", "云达不莱梅"),
        "云达不莱梅": ("Werder Bremen", "云达不莱梅"),

        "Wolfsburg": ("VfL Wolfsburg", "沃尔夫斯堡"),
        "沃尔夫斯堡": ("VfL Wolfsburg", "沃尔夫斯堡"),

        "Stuttgart": ("VfB Stuttgart", "斯图加特"),
        "斯图加特": ("VfB Stuttgart", "斯图加特"),

        "Augsburg": ("FC Augsburg", "奥格斯堡"),
        "奥格斯堡": ("FC Augsburg", "奥格斯堡"),

        "RB Leipzig": ("RB Leipzig", "莱比锡红牛"),
        "莱比锡红牛": ("RB Leipzig", "莱比锡红牛"),

        "Hannover": ("Hannover 96", "汉诺威96"),
        "汉诺威96": ("Hannover 96", "汉诺威96"),

        "Bayern Munich": ("Bayern Munich", "拜仁慕尼黑"),
        "拜仁慕尼黑": ("Bayern Munich", "拜仁慕尼黑"),

        "Hamburg": ("Hamburger SV", "汉堡"),
        "汉堡": ("Hamburger SV", "汉堡"),

        "Ein Frankfurt": ("Eintracht Frankfurt", "法兰克福"),
        "法兰克福": ("Eintracht Frankfurt", "法兰克福"),

        "Bielefeld": ("Arminia Bielefeld", "比勒费尔德"),
        "比勒费尔德": ("Arminia Bielefeld", "比勒费尔德"),

        "Darmstadt": ("Darmstadt 98", "达姆施塔特"),
        "达姆施塔特": ("Darmstadt 98", "达姆施塔特"),

        "Greuther Furth": ("Greuther Fürth", "菲尔特"),
        "菲尔特": ("Greuther Fürth", "菲尔特"),

        "Holstein Kiel": ("Holstein Kiel", "基尔"),
        "基尔": ("Holstein Kiel", "基尔"),

        "Nurnberg": ("1. FC Nürnberg", "纽伦堡"),
        "纽伦堡": ("1. FC Nürnberg", "纽伦堡"),

        "Paderborn": ("Paderborn", "帕德博恩"),
        "帕德博恩": ("Paderborn", "帕德博恩"),

        "Union Berlin": ("Union Berlin", "柏林联合"),
        "柏林联合": ("Union Berlin", "柏林联合"),

        "Bochum": ("VfL Bochum", "波鸿"),
        "VfL Bochum": ("VfL Bochum", "波鸿"),
        "波鸿": ("VfL Bochum", "波鸿"),

        "Heidenheim": ("1. FC Heidenheim", "海登海姆"),
        "海登海姆": ("1. FC Heidenheim", "海登海姆"),

        "St Pauli": ("FC St. Pauli", "圣保利"),
        "圣保利": ("FC St. Pauli", "圣保利"),

        "Fortuna Dusseldorf": ("Fortuna Düsseldorf", "杜塞尔多夫"),
        "杜塞尔多夫": ("Fortuna Düsseldorf", "杜塞尔多夫"),

        "Braunschweig": ("Eintracht Braunschweig", "布伦瑞克"),
        "Eintracht Braunschweig": ("Eintracht Braunschweig", "布伦瑞克"),
        "布伦瑞克": ("Eintracht Braunschweig", "布伦瑞克"),

        "Ingolstadt": ("FC Ingolstadt 04", "因戈尔施塔特"),
        "FC Ingolstadt 04": ("FC Ingolstadt 04", "因戈尔施塔特"),
        "因戈尔施塔特": ("FC Ingolstadt 04", "因戈尔施塔特"),

        "Kaiserslautern": ("1. FC Kaiserslautern", "凯泽斯劳滕"),
        "凯泽斯劳滕": ("1. FC Kaiserslautern", "凯泽斯劳滕"),

        "Bielefeld": ("Arminia Bielefeld", "比勒费尔德"),
        "Arminia Bielefeld": ("Arminia Bielefeld", "比勒费尔德"),
        "比勒费尔德": ("Arminia Bielefeld", "比勒费尔德"),

        "Darmstadt": ("Darmstadt 98", "达姆施塔特"),
        "Darmstadt 98": ("Darmstadt 98", "达姆施塔特"),
        "达姆施塔特": ("Darmstadt 98", "达姆施塔特"),

        "Hannover": ("Hannover 96", "汉诺威96"),
        "Hannover 96": ("Hannover 96", "汉诺威96"),
        "汉诺威96": ("Hannover 96", "汉诺威96"),

        "Hertha": ("Hertha Berlin", "柏林赫塔"),
        "Hertha Berlin": ("Hertha Berlin", "柏林赫塔"),
        "柏林赫塔": ("Hertha Berlin", "柏林赫塔"),

        "Wolfsburg": ("VfL Wolfsburg", "沃尔夫斯堡"),
        "VfL Wolfsburg": ("VfL Wolfsburg", "沃尔夫斯堡"),
        "沃尔夫斯堡": ("VfL Wolfsburg", "沃尔夫斯堡"),

        # ===== 赛程CSV完整名称兼容补充 =====
        "Bayern München": ("Bayern Munich", "拜仁慕尼黑"),
        "SC Freiburg": ("SC Freiburg", "弗赖堡"),
        "FSV Mainz 05": ("Mainz 05", "美因茨"),
        "Mainz 05": ("Mainz 05", "美因茨"),
        "Borussia Dortmund": ("Borussia Dortmund", "多特蒙德"),
        "FC Augsburg": ("FC Augsburg", "奥格斯堡"),
        "1. FC Köln": ("FC Koln", "科隆"),
        "1. FC Koln": ("FC Koln", "科隆"),
        "Borussia Mönchengladbach": ("Borussia Mönchengladbach", "门兴格拉德巴赫"),
        "1899 Hoffenheim": ("Hoffenheim", "霍芬海姆"),
        "Bayer Leverkusen": ("Bayer Leverkusen", "勒沃库森"),
        "Eintracht Frankfurt": ("Eintracht Frankfurt", "法兰克福"),
        "VfB Stuttgart": ("VfB Stuttgart", "斯图加特"),
        "FC Schalke 04": ("Schalke 04", "沙尔克04"),
        "Hamburger SV": ("Hamburger SV", "汉堡"),
        "SC Paderborn 07": ("Paderborn", "帕德博恩"),
        "SV Elversberg": ("SV Elversberg", "埃尔弗斯贝格"),
        "埃尔弗斯贝格": ("SV Elversberg", "埃尔弗斯贝格"),
        "1. FC Kaiserslautern": ("1. FC Kaiserslautern", "凯泽斯劳滕"),
        "1. FC Nürnberg": ("1. FC Nürnberg", "纽伦堡"),
        "FC St. Pauli": ("FC St. Pauli", "圣保利"),
        "Fortuna Düsseldorf": ("Fortuna Düsseldorf", "杜塞尔多夫"),
        "Greuther Fürth": ("Greuther Fürth", "菲尔特"),
        "1. FC Köln": ("FC Koln", "科隆"),
    },

    # ==================== 意甲 SER 完整全量（中英文双key兼容） ====================
    "SER": {
        "Atalanta": ("Atalanta", "亚特兰大"),
        "亚特兰大": ("Atalanta", "亚特兰大"),

        "Inter": ("Inter Milan", "国际米兰"),
        "国际米兰": ("Inter Milan", "国际米兰"),

        "Juventus": ("Juventus", "尤文图斯"),
        "尤文图斯": ("Juventus", "尤文图斯"),

        "Fiorentina": ("Fiorentina", "佛罗伦萨"),
        "佛罗伦萨": ("Fiorentina", "佛罗伦萨"),

        "Genoa": ("Genoa", "热那亚"),
        "热那亚": ("Genoa", "热那亚"),

        "Roma": ("AS Roma", "罗马"),
        "罗马": ("AS Roma", "罗马"),

        "Napoli": ("Napoli", "那不勒斯"),
        "那不勒斯": ("Napoli", "那不勒斯"),

        "Milan": ("AC Milan", "AC米兰"),
        "AC Milan": ("AC Milan", "AC米兰"),
        "AC米兰": ("AC Milan", "AC米兰"),

        "Torino": ("Torino", "都灵"),
        "都灵": ("Torino", "都灵"),

        "Bologna": ("Bologna", "博洛尼亚"),
        "博洛尼亚": ("Bologna", "博洛尼亚"),

        "Empoli": ("Empoli", "恩波利"),
        "恩波利": ("Empoli", "恩波利"),

        "Frosinone": ("Frosinone", "弗罗西诺内"),
        "弗罗西诺内": ("Frosinone", "弗罗西诺内"),

        "Cagliari": ("Cagliari", "卡利亚里"),
        "卡利亚里": ("Cagliari", "卡利亚里"),

        "Cremonese": ("Cremonese", "克雷莫纳"),
        "克雷莫纳": ("Cremonese", "克雷莫纳"),

        "Lecce": ("Lecce", "莱切"),
        "莱切": ("Lecce", "莱切"),

        "Monza": ("Monza", "蒙扎"),
        "蒙扎": ("Monza", "蒙扎"),

        "Parma": ("Parma", "帕尔马"),
        "帕尔马": ("Parma", "帕尔马"),

        "Sassuolo": ("Sassuolo", "萨索洛"),
        "萨索洛": ("Sassuolo", "萨索洛"),

        "Spezia": ("Spezia", "斯佩齐亚"),
        "斯佩齐亚": ("Spezia", "斯佩齐亚"),

        "Udinese": ("Udinese", "乌迪内斯"),
        "Zudda": ("Udinese", "乌迪内斯"),
        "乌迪内斯": ("Udinese", "乌迪内斯"),

        "Venezia": ("Venezia", "威尼斯"),
        "威尼斯": ("Venezia", "威尼斯"),

        "Verona": ("Hellas Verona", "维罗纳"),
        "Hellas Verona": ("Hellas Verona", "维罗纳"),
        "维罗纳": ("Hellas Verona", "维罗纳"),

        "Lazio": ("Lazio", "拉齐奥"),
        "拉齐奥": ("Lazio", "拉齐奥"),

        "Benevento": ("Benevento", "贝内文托"),
        "贝内文托": ("Benevento", "贝内文托"),

        "Chievo": ("Chievo", "切沃"),
        "切沃": ("Chievo", "切沃"),

        "Salernitana": ("Salernitana", "萨勒尼塔纳"),
        "萨勒尼塔纳": ("Salernitana", "萨勒尼塔纳"),

        "Sampdoria": ("Sampdoria", "桑普多利亚"),
        "桑普多利亚": ("Sampdoria", "桑普多利亚"),

        "Brescia": ("Brescia", "布雷西亚"),
        "布雷西亚": ("Brescia", "布雷西亚"),

        "Crotone": ("Crotone", "克罗托内"),
        "克罗托内": ("Crotone", "克罗托内"),

        "Spal": ("SPAL", "斯帕尔"),
        "斯帕尔": ("SPAL", "斯帕尔"),

        "Como": ("Como", "科莫"),
        "科莫": ("Como", "科莫"),

        "Pisa": ("Pisa", "比萨"),
        "比萨": ("Pisa", "比萨"),

        "Bari": ("SSC Bari", "巴里"),
        "巴里": ("SSC Bari", "巴里"),

        "Carpi": ("Carpi FC", "卡尔皮"),
        "卡尔皮": ("Carpi FC", "卡尔皮"),
        
        "Catania": ("Calcio Catania", "卡塔尼亚"),
        "卡塔尼亚": ("Calcio Catania", "卡塔尼亚"),

        "Cesena": ("AC Cesena", "切塞纳"),
        "切塞纳": ("AC Cesena", "切塞纳"),

        "Livorno": ("AS Livorno", "利沃诺"),
        "利沃诺": ("AS Livorno", "利沃诺"),

        "Novara": ("Novara Calcio", "诺瓦拉"),
        "诺瓦拉": ("Novara Calcio", "诺瓦拉"),

        "Palermo": ("US Città di Palermo", "巴勒莫"),
        "巴勒莫": ("US Città di Palermo", "巴勒莫"),
        
        "Pescara": ("Delfino Pescara 1936", "佩斯卡拉"),
        "佩斯卡拉": ("Delfino Pescara 1936", "佩斯卡拉"),

        "Siena": ("Robur Siena", "锡耶纳"),
        "锡耶纳": ("Robur Siena", "锡耶纳"),

        # ===== 赛程CSV完整名称兼容补充 =====
        "AS Roma": ("AS Roma", "罗马"),
        "Hellas Verona": ("Hellas Verona", "维罗纳"),
        "Inter Milan": ("Inter Milan", "国际米兰"),
        "Novara Calcio": ("Novara Calcio", "诺瓦拉"),
        "SSC Bari": ("SSC Bari", "巴里"),
        "Calcio Catania": ("Calcio Catania", "卡塔尼亚"),
        "Carpi FC": ("Carpi FC", "卡尔皮"),
        "Delfino Pescara 1936": ("Delfino Pescara 1936", "佩斯卡拉"),
        "Robur Siena": ("Robur Siena", "锡耶纳"),
        "US Città di Palermo": ("US Città di Palermo", "巴勒莫"),
    },

    # ==================== 法甲 LIG 完整全量（中英文双key兼容） ====================
    "LIG": {
        "Ajaccio": ("Ajaccio", "阿雅克肖"),
        "阿雅克肖": ("Ajaccio", "阿雅克肖"),

        "Amiens": ("Amiens", "亚眠"),
        "亚眠": ("Amiens", "亚眠"),

        "Angers": ("Angers", "昂热"),
        "昂热": ("Angers", "昂热"),

        "Auxerre": ("Auxerre", "欧塞尔"),
        "欧塞尔": ("Auxerre", "欧塞尔"),

        "Brest": ("Brest", "布雷斯特"),
        "布雷斯特": ("Brest", "布雷斯特"),

        "Caen": ("Caen", "卡昂"),
        "卡昂": ("Caen", "卡昂"),

        "Clermont": ("Clermont", "克莱蒙"),
        "克莱蒙": ("Clermont", "克莱蒙"),

        "Dijon": ("Dijon", "第戎"),
        "第戎": ("Dijon", "第戎"),

        "Guingamp": ("Guingamp", "甘冈"),
        "甘冈": ("Guingamp", "甘冈"),

        "Lens": ("Lens", "朗斯"),
        "朗斯": ("Lens", "朗斯"),

        "Le Havre": ("Le Havre", "勒阿弗尔"),
        "勒阿弗尔": ("Le Havre", "勒阿弗尔"),

        "Lille": ("Lille", "里尔"),
        "里尔": ("Lille", "里尔"),

        "Lorient": ("Lorient", "洛里昂"),
        "洛里昂": ("Lorient", "洛里昂"),

        "Lyon": ("Lyon", "里昂"),
        "里昂": ("Lyon", "里昂"),

        "Marseille": ("Marseille", "马赛"),
        "马赛": ("Marseille", "马赛"),

        "Metz": ("Metz", "梅斯"),
        "梅斯": ("Metz", "梅斯"),

        "Montpellier": ("Montpellier", "蒙彼利埃"),
        "蒙彼利埃": ("Montpellier", "蒙彼利埃"),

        "Nantes": ("Nantes", "南特"),
        "南特": ("Nantes", "南特"),

        "Nice": ("Nice", "尼斯"),
        "尼斯": ("Nice", "尼斯"),

        "Nimes": ("Nîmes", "尼姆"),
        "尼姆": ("Nîmes", "尼姆"),

        "Paris SG": ("Paris Saint-Germain", "巴黎圣日耳曼"),
        "Paris Saint-Germain": ("Paris Saint-Germain", "巴黎圣日耳曼"),
        "PSG": ("Paris Saint-Germain", "巴黎圣日耳曼"),
        "巴黎圣日耳曼": ("Paris Saint-Germain", "巴黎圣日耳曼"),

        "Reims": ("Reims", "兰斯"),
        "兰斯": ("Reims", "兰斯"),

        "Rennes": ("Rennes", "雷恩"),
        "雷恩": ("Rennes", "雷恩"),

        "Strasbourg": ("Strasbourg", "斯特拉斯堡"),
        "斯特拉斯堡": ("Strasbourg", "斯特拉斯堡"),

        "St Etienne": ("Saint-Étienne", "圣埃蒂安"),
        "圣埃蒂安": ("Saint-Étienne", "圣埃蒂安"),

        "Toulouse": ("Toulouse", "图卢兹"),
        "图卢兹": ("Toulouse", "图卢兹"),

        "Troyes": ("Troyes", "特鲁瓦"),
        "特鲁瓦": ("Troyes", "特鲁瓦"),

        "Bordeaux": ("Bordeaux", "波尔多"),
        "波尔多": ("Bordeaux", "波尔多"),

        "Paris FC": ("Paris FC", "巴黎FC"),
         "巴黎FC": ("Paris FC", "巴黎FC"),

        "Monaco": ("AS Monaco", "摩纳哥"),
        "摩纳哥": ("AS Monaco", "摩纳哥"),

        "Ajaccio GFCO": ("GFC Ajaccio", "阿雅克肖GFCO"),
        "阿雅克肖GFCO": ("GFC Ajaccio", "阿雅克肖GFCO"),

        "Arles": ("AC Arles-Avignon", "阿尔勒"),
        "阿尔勒": ("AC Arles-Avignon", "阿尔勒"),

        "Bastia": ("SC Bastia", "巴斯蒂亚"),
        "巴斯蒂亚": ("SC Bastia", "巴斯蒂亚"),

        "Boulogne": ("US Boulogne", "布洛涅"),
        "布洛涅": ("US Boulogne", "布洛涅"),

        "Evian Thonon Gaillard": ("Evian Thonon Gaillard FC", "埃维昂"),
        "埃维昂": ("Evian Thonon Gaillard FC", "埃维昂"),
        
        "Grenoble": ("GF38 Grenoble", "格勒诺布尔"),
        "格勒诺布尔": ("GF38 Grenoble", "格勒诺布尔"),

        "Nancy": ("AS Nancy Lorraine", "南锡"),
        "AS Nancy Lorraine": ("AS Nancy Lorraine", "南锡"),
        "南锡": ("AS Nancy Lorraine", "南锡"),

        "Sochaux": ("FC Sochaux-Montbéliard", "索肖"),
        "FC Sochaux-Montbéliard": ("FC Sochaux-Montbéliard", "索肖"),
        "索肖": ("FC Sochaux-Montbéliard", "索肖"),
        
        "Valenciennes": ("Valenciennes FC", "瓦朗谢讷"),
        "Valenciennes FC": ("Valenciennes FC", "瓦朗谢讷"),
        "瓦朗谢讷": ("Valenciennes FC", "瓦朗谢讷"),

        "Bastia": ("SC Bastia", "巴斯蒂亚"),
        "SC Bastia": ("SC Bastia", "巴斯蒂亚"),
        "巴斯蒂亚": ("SC Bastia", "巴斯蒂亚"),

        "Brest": ("Brest", "布雷斯特"),
        "Stade Brestois 29": ("Brest", "布雷斯特"),
        "布雷斯特": ("Brest", "布雷斯特"),

        "Troyes": ("Troyes", "特鲁瓦"),
        "ESTAC Troyes": ("Troyes", "特鲁瓦"),
        "特鲁瓦": ("Troyes", "特鲁瓦"),

        # ===== 赛程CSV完整名称兼容补充 =====
        "Paris Saint Germain": ("Paris Saint-Germain", "巴黎圣日耳曼"),
        "Stade Brestois 29": ("Brest", "布雷斯特"),
        "Estac Troyes": ("Troyes", "特鲁瓦"),
        "Le Mans": ("Le Mans", "勒芒"),
        "勒芒": ("Le Mans", "勒芒"),

        "Evian Thonon Gaillard FC": ("Evian Thonon Gaillard FC", "埃维昂"),
        "GF38 Grenoble": ("GF38 Grenoble", "格勒诺布尔"),
        "GFC Ajaccio": ("GFC Ajaccio", "阿雅克肖GFCO"),
        "Le Mans": ("Le Mans", "勒芒"),
        "US Boulogne": ("US Boulogne", "布洛涅"),
        "AC Arles-Avignon": ("AC Arles-Avignon", "阿尔勒"),
        "Saint-Étienne": ("Saint-Étienne", "圣埃蒂安"),
    }
}

# 【2026-08-08 补充：历史遗留英文队名映射】match_feature_final 球队列中英混杂（约80%中文、20%英文），
#  以下 27 支是数据中直接以英文存储、原映射表缺失的队名（看板比赛分析/球队详情/预测双向匹配会用到），
#  以数据里的确切写法为 key，追加进对应联赛 dict（不覆盖已有映射）。
_LEGACY_MISSING_TEAMS = {
    "BUN": {
        "MGladbach": ("Borussia M'gladbach", "门兴格拉德巴赫"),
        "Cottbus": ("Energie Cottbus", "科特布斯"),
        "Hansa Rostock": ("FC Hansa Rostock", "罗斯托克"),
        "Munich 1860": ("TSV 1860 München", "慕尼黑1860"),
        "Karlsruhe": ("Karlsruher SC", "卡尔斯鲁厄"),
        "Duisburg": ("MSV Duisburg", "杜伊斯堡"),
        "Aachen": ("Alemannia Aachen", "亚琛"),
        "Unterhaching": ("SpVgg Unterhaching", "翁特哈兴"),
    },
    "SER": {
        "Reggina": ("Reggina", "雷吉纳"),
        "Messina": ("Messina", "墨西拿"),
        "Perugia": ("Perugia", "佩鲁贾"),
        "Ascoli": ("Ascoli", "阿斯科利"),
        "Piacenza": ("Piacenza", "皮亚琴察"),
        "Modena": ("Modena", "摩德纳"),
        "Treviso": ("Treviso", "特雷维索"),
        "Vicenza": ("Vicenza", "维琴察"),
        "Ancona": ("Ancona", "安科纳"),
    },
    "EPL": {
        "Charlton": ("Charlton Athletic", "查尔顿"),
        "Derby": ("Derby County", "德比郡"),
        "Nottm Forest": ("Nottingham Forest", "诺丁汉森林"),
        "Bradford": ("Bradford City", "布拉德福德"),
    },
    "LLA": {
        "Recreativo": ("Recreativo Huelva", "维尔瓦"),
        "Numancia": ("Numancia", "努曼西亚"),
        "Murcia": ("Real Murcia", "穆尔西亚"),
        "Albacete": ("Albacete", "阿尔瓦塞特"),
    },
    "LIG": {
        "Sedan": ("CS Sedan", "色当"),
        "Istres": ("Istres FC", "伊斯特尔"),
    },
}
for _lg, _m in _LEGACY_MISSING_TEAMS.items():
    LEAGUE_TEAM_MAP.setdefault(_lg, {})
    for _k, _v in _m.items():
        if _k not in LEAGUE_TEAM_MAP[_lg]:
            LEAGUE_TEAM_MAP[_lg][_k] = _v

# 联赛基础信息
LEAGUE_CFG = {
    "EPL": {"name": "英超", "start_year": 2018},
    "LLA": {"name": "西甲", "start_year": 2018},
    "BUN": {"name": "德甲", "start_year": 2018},
    "SER": {"name": "意甲", "start_year": 2018},
    "LIG": {"name": "法甲", "start_year": 2018}
}

# ========== 联赛编码统一转换（配置键 ↔ 数据库原始编码） ==========
# 【2026-08-05 统一】全部从 common_config.LEAGUE_REGISTRY 派生，全项目唯一编码出口。
# 配置键（前端下拉用）：EPL / BUN / LLA / SER / LIG
# 数据库编码（表内实际存储）：E0 / D1 / SP1 / I1 / F1
# ⚠️ 旧硬编码误把 西甲=LLA、意甲=SER、法甲=LIG（那是旧 match_result 的翻译码），
#    与真实库码 SP1/I1/F1 冲突，曾导致联赛独热大面积失效、full_update_data 写错库码。
from common_config import LEAGUE_REGISTRY
CFG_2_DB_CODE = {k: v["db_code"] for k, v in LEAGUE_REGISTRY.items()}
DB_CODE_2_CFG = {v["db_code"]: k for k, v in LEAGUE_REGISTRY.items()}


def cfg_to_db_league(cfg_key):
    """配置键 → 数据库联赛编码"""
    return CFG_2_DB_CODE.get(cfg_key, cfg_key)


def db_to_cfg_league(db_code):
    """数据库联赛编码 → 配置键"""
    return DB_CODE_2_CFG.get(db_code, db_code)


def get_standard_team(league_code, raw_name):
    """
    统一标准化队名【2026-07-30 生产级】
    - 精确匹配（原逻辑保留）
    - 归一化匹配（变音 / 前缀 / 全称 / 简写）
    - 关闭跨联赛兜底（保持你 7-24 的设计）
    """
    if not league_code or not isinstance(raw_name, str) or raw_name.strip() == "":
        return (raw_name, raw_name)

    raw_name = raw_name.strip()
    league_dict = LEAGUE_TEAM_MAP.get(league_code, {})

    # 1️⃣ 原样精确匹配（最快）
    if raw_name in league_dict:
        return league_dict[raw_name]

    # 2️⃣ 归一化匹配（解决变音、全称/简称问题）
    norm_raw = _norm_team_name(raw_name)
    for k, v in league_dict.items():
        if _norm_team_name(k) == norm_raw:
            return v

    # 3️⃣ 兜底（保持你原设计）
    return (raw_name, raw_name)


def get_team_cn_name_v2(league_raw, raw_team_name, print_miss: bool = True):
    """
    对外中文队名统一接口（适配数据库真实联赛编码）
    数据库：E0/D1/SP1/I1/F1
    配置键：EPL/BUN/LLA/SER/LIG
    先按联赛精准匹配，关闭跨联赛兜底
    2026-07-23 新增空值过滤、缺失映射日志打印
    2026-07-24 20:30 新增print_miss开关，批量转换时关闭打印，仅统计校验开启打印
    """
    # 空值直接返回原始名称，避免报错
    if league_raw is None or raw_team_name is None or str(raw_team_name).strip() == "":
        return str(raw_team_name) if raw_team_name is not None else ""
    
    # 【2026-08-05 统一】同时兼容真实库码(SP1/I1/F1/E0/D1)与配置键(EPL/BUN/LLA/SER/LIG)
    league_map = dict(DB_CODE_2_CFG)      # E0→EPL, D1→BUN, SP1→LLA, I1→SER, F1→LIG
    for _k in LEAGUE_TEAM_MAP:
        league_map.setdefault(_k, _k)     # 直接传配置键也能命中
    league_key = league_map.get(str(league_raw).strip(), str(league_raw).strip())
    std_name, cn_name = get_standard_team(league_key, raw_team_name.strip())
    
    # 仅print_miss=True时才打印终端缺失日志
    if std_name == raw_team_name.strip() and cn_name == raw_team_name.strip() and print_miss:
        print(f"【球队映射缺失】联赛编码:{league_raw},配置键:{league_key},原始队名:{raw_team_name}")
    return cn_name

def build_cn_to_std_map():
    """2026-07-24 优化：避免不同联赛中文覆盖，先入为主不重复覆盖"""
    cn2std = {}
    league_order = ["EPL", "LLA", "BUN", "SER", "LIG"]
    for league in league_order:
        league_dict = LEAGUE_TEAM_MAP[league]
        for _, (std_name, cn_name) in league_dict.items():
            if cn_name not in cn2std:
                cn2std[cn_name] = std_name
    return cn2std

# 全局双向映射字典，供看板复用
cn_2_std = build_cn_to_std_map()