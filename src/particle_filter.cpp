/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

#define EPS 1e-6

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  for (int i=0; i<num_particles; i++) {
    Particle particle = particles[i];

    double new_theta = particle.theta + yaw_rate * delta_t;
    double velocity_to_yaw_rate = fabs(yaw_rate) < EPS ? 0 : velocity/yaw_rate; // todo: float comparison
    double new_x = particle.x + velocity_to_yaw_rate * (sin(new_theta) - sin(particle.theta));
    double new_y = particle.y + velocity_to_yaw_rate * (cos(particle.theta) - cos(new_theta));
    
    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  weights.clear();
  for (int p = 0; p < num_particles; p++) {
    vector<LandmarkObs> mappedObservations = mapObservations(particles[p], observations);
    vector<LandmarkObs> landmarksInRange = getLandmarksInRange(particles[p], map_landmarks, sensor_range);
    dataAssociation(landmarksInRange, mappedObservations);
    particles[p].weight = calculateWeight(mappedObservations, landmarksInRange, std_landmark);
    weights.push_back(particles[p].weight);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
    for (int i=0; i<observations.size(); i++) {
      LandmarkObs obs = observations[i];
      
      int closest_landmark_id = -1;
      double min_dist = 999999.99;

      for (int j = 0; j < predicted.size(); j++) {
        LandmarkObs landmark = predicted[j];
        double dx = obs.x - landmark.x;
        double dy = obs.y - landmark.y;
        double curr_dist = sqrt(dx * dx + dy * dy);
        if (curr_dist < min_dist) {
          min_dist = curr_dist;
          closest_landmark_id = landmark.id;
        }
      }

      observations[i].id = closest_landmark_id;
    }
}

vector<LandmarkObs> ParticleFilter::mapObservations(Particle particle,
                                          const vector<LandmarkObs>& observations) {
  vector<LandmarkObs> mappedObservations;
  
  for (int i=0; i<observations.size(); i++) {
    LandmarkObs obs = observations[i];
    double x_map = particle.x + (cos(particle.theta) * obs.x) - (sin(particle.theta) * obs.y);
    double y_map = particle.y + (sin(particle.theta) * obs.x) + (cos(particle.theta) * obs.y);
    mappedObservations.push_back(LandmarkObs{ obs.id, x_map, y_map});
  }

  return mappedObservations;
}

vector<LandmarkObs> ParticleFilter::getLandmarksInRange(Particle particle, const Map &map_landmarks, double sensor_range) {
  vector<LandmarkObs> landmarksInRange;
  for (int l=0; l<map_landmarks.landmark_list.size(); l++) {
    float landmark_x = map_landmarks.landmark_list[l].x_f;
    float landmark_y = map_landmarks.landmark_list[l].y_f;
    int id = map_landmarks.landmark_list[l].id_i;
    double dx = particle.x - landmark_x;
    double dy = particle.y - landmark_y;
    if (sqrt(dx*dx + dy*dy) <= sensor_range) {
      landmarksInRange.push_back(LandmarkObs{ id, landmark_x, landmark_y });
    }
  }
  return landmarksInRange;
}

double ParticleFilter::calculateWeight(vector<LandmarkObs> mappedObservations, vector<LandmarkObs> inrangeLandmarks, double std_landmark[]) {
  double weight = 1.0;
  
  for (int o=0; o<mappedObservations.size(); o++) {
    LandmarkObs observation = mappedObservations[o];

    std::vector<LandmarkObs>::iterator closest_landmark = std::find_if(inrangeLandmarks.begin(), inrangeLandmarks.end(), [&observation](const LandmarkObs & val){
                                              return val.id == observation.id;
                                            });

    double noise = std_landmark[0]*std_landmark[1];
    double gauss_norm = 1/(2*M_PI*noise);
    double dx = observation.x - closest_landmark->x;
    double dy = observation.y - closest_landmark->y;
    double exponent = dx*dx/(2*noise) + dy*dy/(2*noise);
    double obsWeight =  gauss_norm * exp(-exponent);

    if (obsWeight < EPS) {
      weight *= EPS;
    } else {
      weight *= obsWeight;
    }
  }  

  return weight;
}

void ParticleFilter::resample() {
  std::uniform_int_distribution<int> dist_particles(0, num_particles - 1);
  int index = dist_particles(gen);

  double max_weight = *max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> dist_weight(0.0, max_weight);

  vector<Particle> result;
  double beta = 0.0;
  for (int i=0; i<num_particles; i++) {
    beta +=  dist_weight(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index+1) % num_particles;
    }
    result.push_back(particles[index]);
  }

  particles = result;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}