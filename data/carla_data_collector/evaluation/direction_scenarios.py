"""Real CARLA mixed-direction tests on an explicitly generated straight road.

Run sequentially, never beside another simulator test. Restores the prior map.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'data/carla_data_collector'))
import carla
import yaml
from evaluation.route_recovery_scenarios import run_case


def road_xml(solid=False, both_same=False):
    def lane_xml(lane_id, mark):
        return ('<lane id="{}" type="driving" level="false"><link/>'
                '<width sOffset="0" a="3.5" b="0" c="0" d="0"/>'
                '<roadMark sOffset="0" type="{}" weight="standard" color="white" '
                'width="0.12" laneChange="both"/></lane>').format(lane_id, mark)
    left = '' if both_same else '<left>' + lane_xml(1, 'solid') + '</left>'
    right = ''.join(lane_xml(i, 'broken' if i > (-3 if both_same else -2) else 'solid')
                    for i in range(-1, -4 if both_same else -3, -1))
    return ('<?xml version="1.0" encoding="UTF-8"?><OpenDRIVE>'
            '<header revMajor="1" revMinor="4" name="direction-stage1" version="1" '
            'date="2026-10-25" north="0" south="0" east="0" west="0"/>'
            '<road name="direction-test" length="900" id="1" junction="-1"><link/>'
            '<type s="0" type="town"><speed max="30" unit="km/h"/></type>'
            '<planView><geometry s="0" x="0" y="0" hdg="0" length="900"><line/></geometry></planView>'
            '<elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>'
            '<lateralProfile/><lanes><laneSection s="0">' + left +
            '<center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="' +
            ('solid solid' if solid else 'broken') +
            '" weight="standard" color="yellow" width="0.12" laneChange="both"/></lane></center>'
            '<right>' + right + '</right></laneSection></lanes></road></OpenDRIVE>')


CASES = (
    ('both_same_direction', False, True, False, 'none', 12),
    ('mixed_prefer_same', False, False, False, 'none', 0),
    ('mixed_opposing_clear', False, False, True, 'none', 0),
    ('mixed_opposing_oncoming', False, False, True, 'oncoming', 12),
    ('same_direction_lane_wrongway', False, True, True, 'oncoming', 12),
    ('mixed_solid_border', True, False, True, 'none', 0),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.)
    original_world = client.get_world()
    previous = original_world.get_map().name.split('/')[-1]
    original_settings, original_weather = original_world.get_settings(), original_world.get_weather()
    if previous == 'OpenDriveMap':
        previous = 'Town05_Opt'
    config = yaml.safe_load((ROOT / 'config/default.yaml').read_text(encoding='utf-8'))['carla_collector']['ego']['behavior_agent']
    try:
        for name, solid, both_same, blocked, mode, density in CASES:
            if args.cases and name not in args.cases:
                continue
            print('scenario', name, flush=True)
            random.seed(0)
            xml = road_xml(solid, both_same)
            world = client.generate_opendrive_world(xml, carla.OpendriveGenerationParameters(
                vertex_distance=2., wall_height=0., additional_width=.0, smooth_junctions=False,
                enable_mesh_visibility=True))
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = .1
            settings.no_rendering_mode = True
            world.apply_settings(settings)
            world.set_weather(carla.WeatherParameters.ClearNoon)
            world.tick()
            start = world.get_map().get_waypoint_xodr(1, -2 if both_same else -1, 100.)
            result = run_case(world, start, config, 'avoid_left', 180 if solid else 650, .55,
                              mode, density, -1, blocked, True)
            result.update(seed=0, direction_scenario=name, opendrive_xml=xml)
            if solid:
                result['expected_outcome'] = 'wait'
            path = args.output / 'records' / (name + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(name, 'frames', result['frames'], 'goal', result['goal_reached'],
                  'collisions', len(result['collisions']), 'states', result['states'], flush=True)
    finally:
        restored = client.load_world(previous)
        restored.apply_settings(original_settings)
        restored.set_weather(original_weather)


if __name__ == '__main__':
    main()
