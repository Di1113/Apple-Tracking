# vim: expandtab:ts=4:sw=4
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric,max_age=75, n_init=1,v=np.array([0,0,20]),mean=np.array([0,0,0]),_st_weight_position=2):
        self.metric = metric
        # self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter(v,_st_weight_position)
        self.tracks = Track(initialized=False)
        self._next_id = np.array([],dtype=np.uint16)
        self.mean_3D=mean
        self.covariance_3D=np.eye(3,3)*_st_weight_position
        self.mean_2D=np.array([[0,0,0,0]])
        self.confidence=np.array([0])

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """

        self.tracks.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : object consisting of all the apples and rover measurements

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # import ipdb;ipdb.set_trace()
        # Update track set.
        if len(matches)!=0:
            self.tracks.update(self.kf, detections,matches)

        for track_idx in unmatched_tracks:
            self.tracks.mark_missed(track_idx)

        if len(unmatched_detections)!=0:
            self._initiate_track(detections,unmatched_detections)
        # MATCHED_TRACKS=[self.tracks[i] for i in unmatched_tracks]
        # UNMACTHED_DETECTIONS=[detections[i] for i in unmatched_detections]
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        return matches

    def _match(self, detections):

        '''
        detections: object of class detection
        '''

        def gated_metric(tracks, dets, track_indices):
            # import ipdb; ipdb.set_trace()
            detection = dets.points_3D
            cost_matrix = self.metric.distance(detection, tracks, track_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # import ipdb;ipdb.set_trace()

        
        # confirmed_tracks = [
        #     i for i in range(len(self.tracks.state)) if self.tracks.is_confirmed(i)]
        # unconfirmed_tracks = [
        #     i for i in range(len(self.tracks.state)) if not self.tracks.is_confirmed(i)] #Unmatched tracks from previous frame
        # import ipdb; ipdb.set_trace()

        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections = linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections)
        # import ipdb; ipdb.set_trace()
        # unmatched_tracks = list(set(unmatched_tracks + unconfirmed_tracks))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, unmatched_detections):
        '''
        detection: object of class detection consisting of all the detections
        unmactched_detection: list consisting of indices of bbox of image space that are unmatched detections, starting from pos 1
        '''
        
        mean, covariance = self.kf.initiate(detection.points_3D,unmatched_detections)
        # import ipdb; ipdb.set_trace()        
        #merging existing covaraince with new detection covariance
        self.covariance_3D=np.vstack((np.hstack((self.covariance_3D, np.zeros((len(self.covariance_3D),len(covariance[0]))))),
                                    np.hstack((np.zeros((len(covariance),len(self.covariance_3D[0]))),covariance))))
        self.mean_3D=np.vstack((self.mean_3D,mean))
        self.mean_2D=np.vstack((self.mean_2D,detection.points_2D[unmatched_detections]))
        self.confidence=np.concatenate((self.confidence,detection.confidence[unmatched_detections]),axis=0)
        self._next_id=np.arange(len(self.confidence))
        class_name = detection.get_class()

        self.tracks=Track(
            mean_2D=self.mean_2D, mean_3D=self.mean_3D,covariance_3D=self.covariance_3D, track_id=self._next_id, n_init=self.n_init,
             max_age=np.ones(len(self.confidence))*self.max_age,class_name=class_name,confidence=self.confidence,initialized=True)
