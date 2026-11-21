"""Per-cache settings, separate from the individual crystal records."""
import json
from unifiedgroupgen.pl_data.cache import write_tensor_cache_config


def test_cache_config_records_effective_settings_beside_each_cache(tmp_path):
    first = tmp_path / 'train.first.pt'
    second = tmp_path / 'train.second.pt'
    source = 'data/src/fixture/train.csv'
    identity = dict(dataset='fixture', split='train', conversion=dict(symmetry_tolerance=.1),
                    graph_options=dict(cutoff=6., max_neighbors=64, max_images=16))
    first.write_bytes(b'cache')
    config = write_tensor_cache_config(first, identity, source, 7, mode='csv', workers=2,
                                       row_wait_timeout_seconds=120)
    assert config == tmp_path / 'train.first.config.json'
    report = json.loads(config.read_text(encoding='utf-8'))
    assert report['cache'] == first.name and report['source_csv'] == source
    assert report['identity'] == identity and report['records'] == 7
    assert report['build']['row_wait_timeout_seconds'] == 120
    assert report['graph_policy']['keep_equal_distance_shells']
    prior = config.read_bytes()
    identity['conversion']['symmetry_tolerance'] = .01
    write_tensor_cache_config(second, identity, source, 7, mode='csv')
    assert config.read_bytes() == prior and first.read_bytes() == b'cache'
