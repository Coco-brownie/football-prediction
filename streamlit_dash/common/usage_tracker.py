"""
使用埋点统计模块
自动记录用户操作行为，用于产品优化分析
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'football.db')


def init_table():
    """初始化埋点表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            action_type TEXT,
            action_detail TEXT,
            page_name TEXT,
            ip_address TEXT,
            extra TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_time ON usage_log(log_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_action ON usage_log(action_type)")
    conn.close()


def track(action_type, action_detail='', page_name='', session_id='', ip='', extra=''):
    """
    记录一条用户行为埋点

    Args:
        action_type: 操作类型（page_view / predict / filter / search 等）
        action_detail: 操作详情（如预测了哪场比赛）
        page_name: 所在页面
        session_id: 会话标识
        ip: IP地址（可选）
        extra: 额外信息JSON
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO usage_log (action_type, action_detail, page_name, session_id, ip_address, extra) VALUES (?,?,?,?,?,?)",
            (action_type, action_detail, page_name, session_id, ip, extra)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # 埋点失败不影响主流程
        pass


def get_stats(days=7):
    """获取最近N天的使用统计"""
    conn = sqlite3.connect(DB_PATH)

    # 总访问量
    cur = conn.execute(
        "SELECT COUNT(*) FROM usage_log WHERE log_time >= datetime('now', ?)",
        (f'-{days} days',)
    )
    total = cur.fetchone()[0]

    # 各操作类型分布
    cur = conn.execute(
        "SELECT action_type, COUNT(*) as cnt FROM usage_log WHERE log_time >= datetime('now', ?) GROUP BY action_type ORDER BY cnt DESC",
        (f'-{days} days',)
    )
    actions = cur.fetchall()

    # 各页面访问量
    cur = conn.execute(
        "SELECT page_name, COUNT(*) as cnt FROM usage_log WHERE log_time >= datetime('now', ?) AND page_name != '' GROUP BY page_name ORDER BY cnt DESC",
        (f'-{days} days',)
    )
    pages = cur.fetchall()

    # 每日活跃趋势
    cur = conn.execute("""
        SELECT date(log_time) as dt, COUNT(*) as cnt 
        FROM usage_log 
        WHERE log_time >= datetime('now', ?)
        GROUP BY date(log_time) 
        ORDER BY dt DESC
    """, (f'-{days} days',))
    daily = cur.fetchall()

    conn.close()
    return {
        'total': total,
        'actions': actions,
        'pages': pages,
        'daily': daily,
    }


# 初始化表
init_table()
